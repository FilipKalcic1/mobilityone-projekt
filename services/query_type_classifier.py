"""
Query Type Classifier - Klasificira upit PRIJE FAISS pretrage.

Ovo je KRITIČNA komponenta koja rješava problem:
- get_Companies_id vs get_Companies_id_documents vs get_Companies_id_metadata
  imaju 0.90+ similarity, ali su POTPUNO različiti alati.

Rješenje: Klasificirati upit u tip PRIJE pretrage, pa filtrirati po sufiksu.

Tipovi upita:
- SINGLE_ENTITY: "dohvati kompaniju X" → preferira _id suffix
- DOCUMENTS: "dokumenti kompanije" → preferira _documents suffix
- METADATA: "struktura podataka" → preferira _metadata suffix
- LIST: "sve kompanije" → preferira bazni endpoint bez sufiksa
- AGGREGATION: "ukupno troškova" → preferira _Agg, _GroupBy
- TREE: "hijerarhija" → preferira _tree suffix
- DELETE_CRITERIA: "obriši sve stare" → preferira _DeleteByCriteria
- BULK_UPDATE: "ažuriraj sve" → preferira _multipatch
- DEFAULT_SET: "postavi kao zadano" → preferira _SetAsDefault
- THUMBNAIL: "slika", "preview" → preferira _thumb
- PROJECTION: "samo ime i email" → preferira _ProjectTo
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Tipovi upita za suffix filtering."""
    SINGLE_ENTITY = "single_entity"      # Dohvati jednu stavku po ID-u
    DOCUMENTS = "documents"               # Rad s dokumentima/prilozima
    METADATA = "metadata"                 # Metapodaci, struktura, shema
    LIST = "list"                         # Lista svih stavki
    AGGREGATION = "aggregation"           # Agregacije, statistike
    TREE = "tree"                         # Hijerarhijska struktura
    DELETE_CRITERIA = "delete_criteria"   # Brisanje po kriterijima
    BULK_UPDATE = "bulk_update"           # Bulk/batch ažuriranje
    DEFAULT_SET = "default_set"           # Postavljanje kao zadano
    THUMBNAIL = "thumbnail"               # Sličice, preview
    PROJECTION = "projection"             # Projekcija određenih polja
    UNKNOWN = "unknown"                   # Nije prepoznato


@dataclass
class QueryTypeResult:
    """Rezultat klasifikacije upita."""
    query_type: QueryType
    confidence: float
    matched_pattern: str
    preferred_suffixes: List[str]      # Sufiksi koji se preferiraju
    excluded_suffixes: List[str]       # Sufiksi koji se isključuju


# Definicija tipova upita s patternima i suffix pravilima
QUERY_TYPE_DEFINITIONS = {
    QueryType.DOCUMENTS: {
        "patterns": [
            (r"\bdokument", 0.95),
            (r"\bprilog", 0.95),
            (r"\bdatoteka", 0.95),
            (r"\bupload", 0.90),
            (r"\bpreuzmi\s+dokument", 0.95),
            (r"\bdodaj\s+prilog", 0.95),
            (r"\bprilozi\b", 0.90),
            (r"\bdokumentacij", 0.90),
            (r"\bpdf\b", 0.85),
            (r"\bslika\s+dokument", 0.90),
        ],
        "preferred_suffixes": ["_id_documents_documentId", "_id_documents", "_documents"],
        "excluded_suffixes": ["_metadata", "_Agg", "_GroupBy", "_tree"]
    },

    QueryType.THUMBNAIL: {
        "patterns": [
            (r"\bthumb", 0.98),
            (r"\bsličic", 0.98),  # Higher than documents
            (r"\bpreview", 0.92),
            (r"\bmala\s+slika", 0.92),
            (r"\bikona\b", 0.88),
            (r"\bpregled\s+slike", 0.92),
        ],
        "preferred_suffixes": ["_thumb", "_id_documents_documentId_thumb"],
        "excluded_suffixes": ["_metadata", "_Agg"]
    },

    QueryType.METADATA: {
        "patterns": [
            (r"\bmetapodac", 0.95),
            (r"\bstruktur", 0.90),
            (r"\bshem", 0.90),
            (r"\bpolj", 0.85),
            (r"\bkolon", 0.85),
            (r"\bdefinicij", 0.85),
            (r"\batribut", 0.85),
            (r"\bkonfigur", 0.80),
        ],
        "preferred_suffixes": ["_id_metadata", "_metadata", "_Metadata"],
        "excluded_suffixes": ["_documents", "_thumb", "_Agg"]
    },

    QueryType.AGGREGATION: {
        "patterns": [
            (r"\bukupno", 0.95),
            (r"\bsum[ae]", 0.95),
            (r"\bprosjek", 0.95),
            (r"\bstatistik", 0.95),
            (r"\bbroj\s+\w+", 0.85),
            (r"\bkoliko\s+ima", 0.90),
            (r"\bizvješt", 0.80),
            (r"\bagregir", 0.95),
            (r"\bgrupiraj", 0.90),
            (r"\bpo\s+mjesec", 0.85),
            (r"\bpo\s+godin", 0.85),
        ],
        "preferred_suffixes": ["_Agg", "_GroupBy", "_Aggregation"],
        "excluded_suffixes": ["_id", "_documents", "_metadata"]
    },

    QueryType.TREE: {
        "patterns": [
            (r"\bhijerarhij", 0.95),
            (r"\bstabl", 0.95),
            (r"\btree\b", 0.95),
            (r"\bparent", 0.90),
            (r"\bchild", 0.90),
            (r"\broditelj", 0.90),
            (r"\bdijete", 0.85),
            (r"\bpodređen", 0.90),
            (r"\bnadređen", 0.90),
        ],
        "preferred_suffixes": ["_tree"],
        "excluded_suffixes": ["_documents", "_metadata", "_Agg"]
    },

    QueryType.DELETE_CRITERIA: {
        "patterns": [
            (r"\bobriši\s+sve", 0.95),
            (r"\bizbriši\s+sve", 0.95),
            (r"\bobriši\s+po\s+kriteri", 0.95),
            (r"\bbulk\s+delete", 0.95),
            (r"\bmasovno\s+briš", 0.95),
            (r"\bobriši\s+gdje", 0.90),
            (r"\bobriši\s+star", 0.85),
        ],
        "preferred_suffixes": ["_DeleteByCriteria"],
        "excluded_suffixes": ["_id", "_documents"]
    },

    QueryType.BULK_UPDATE: {
        "patterns": [
            (r"\bažuriraj\s+sve", 0.95),
            (r"\bbulk\s+update", 0.95),
            (r"\bmasovno\s+ažurir", 0.95),
            (r"\bmultipatch", 0.95),
            (r"\bizmijeni\s+više", 0.90),
            (r"\bpromijeni\s+sve", 0.90),
        ],
        "preferred_suffixes": ["_multipatch", "_bulk"],
        "excluded_suffixes": ["_id", "_documents"]
    },

    QueryType.DEFAULT_SET: {
        "patterns": [
            (r"\bpostavi\s+kao\s+zadan", 0.98),  # Higher than documents
            (r"\bset\s+as\s+default", 0.98),
            (r"\bzadani\b", 0.92),
            (r"\bdefault\b", 0.92),
            (r"\bprimarni\b", 0.88),
            (r"\bglavni\s+dokument", 0.88),
            (r"\boznači\s+kao\s+zadano", 0.98),
        ],
        "preferred_suffixes": ["_SetAsDefault", "_id_documents_documentId_SetAsDefault"],
        "excluded_suffixes": ["_thumb", "_Agg"]
    },

    QueryType.PROJECTION: {
        "patterns": [
            (r"\bsamo\s+\w+\s+i\s+\w+", 0.90),  # "samo ime i email"
            (r"\bprojekci", 0.95),
            (r"\bodređen\w*\s+polj", 0.90),
            (r"\bselect\s+fields", 0.90),
            (r"\bvrati\s+samo", 0.85),
        ],
        "preferred_suffixes": ["_ProjectTo"],
        "excluded_suffixes": ["_documents", "_metadata"]
    },

    QueryType.LIST: {
        "patterns": [
            (r"\bsve\s+\w+e\b", 0.85),     # "sve kompanije", "sve osobe"
            (r"\bsvi\s+\w+i\b", 0.85),     # "svi zaposlenici"
            (r"\bsva\s+\w+a\b", 0.85),     # "sva vozila"
            (r"\blista\b", 0.90),
            (r"\bpopis\b", 0.90),
            (r"\bpregled\s+svi", 0.85),
            (r"\bdohvati\s+sve", 0.90),
            (r"\bprikaži\s+sve", 0.90),
        ],
        "preferred_suffixes": [],  # Bazni endpoint
        "excluded_suffixes": ["_id", "_id_documents", "_id_metadata", "_Agg", "_tree"]
    },

    QueryType.SINGLE_ENTITY: {
        "patterns": [
            (r"\bpo\s+id", 0.95),               # "po ID-u", "po id"
            (r"\bs\s+id-?e?m", 0.95),           # "s ID-em", "s idem"
            (r"\bdohvati\s+\w+\s+\d+", 0.92),   # "dohvati kompaniju 123"
            (r"\bprikaži\s+\w+\s+\d+", 0.92),   # "prikaži vozilo 456"
            (r"\binfo\s+o\s+\w+", 0.88),
            (r"\bdetalji\s+\w+", 0.88),
            (r"\bprikaži\s+detalje", 0.88),    # "prikaži detalje vozila"
            (r"\bpodaci\s+o\s+\w+", 0.88),
            (r"\bjedna\s+\w+", 0.82),
            (r"\bjednog\s+\w+", 0.82),
            (r"\bjedan\s+\w+", 0.82),
            (r"\bkonkretn\w+\s+\w+", 0.88),
            (r"\bodređen\w+\s+\w+", 0.82),
            (r"\bdohvati\s+\w+\b$", 0.75),     # "dohvati kompaniju" (end of string)
        ],
        "preferred_suffixes": ["_id"],
        "excluded_suffixes": ["_documents", "_metadata", "_Agg", "_tree", "_thumb"]
    },
}


class QueryTypeClassifier:
    """
    Klasificira korisničke upite u tipove za bolje filtriranje alata.

    Ovo rješava problem:
    - FAISS vraća get_X_id, get_X_id_documents, get_X_id_metadata sa 0.90+ similarity
    - Ali korisnik želi TOČNO jedan od njih
    - Classifier detektira tip upita i filtrira po sufiksu
    """

    def __init__(self):
        """Inicijalizacija classifiera."""
        self._compiled_patterns: dict = {}
        self._compile_patterns()

    def _compile_patterns(self):
        """Kompilira regex patterne za brzinu."""
        for query_type, definition in QUERY_TYPE_DEFINITIONS.items():
            compiled = []
            for pattern, confidence in definition["patterns"]:
                try:
                    compiled.append((re.compile(pattern, re.IGNORECASE), confidence, pattern))
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern}': {e}")
            self._compiled_patterns[query_type] = compiled

    def classify(self, query: str) -> QueryTypeResult:
        """
        Klasificiraj upit u tip.

        Args:
            query: Korisnički upit

        Returns:
            QueryTypeResult s tipom, confidence i suffix pravilima
        """
        query_lower = query.lower().strip()

        best_match: Optional[Tuple[QueryType, float, str]] = None

        # Prolazi kroz sve tipove i traži najbolji match
        for query_type, patterns in self._compiled_patterns.items():
            for regex, confidence, pattern_str in patterns:
                if regex.search(query_lower):
                    if best_match is None or confidence > best_match[1]:
                        best_match = (query_type, confidence, pattern_str)

        if best_match:
            query_type, confidence, pattern = best_match
            definition = QUERY_TYPE_DEFINITIONS[query_type]

            logger.debug(
                f"QueryTypeClassifier: '{query[:40]}...' -> {query_type.value} "
                f"(conf={confidence:.2f}, pattern='{pattern}')"
            )

            return QueryTypeResult(
                query_type=query_type,
                confidence=confidence,
                matched_pattern=pattern,
                preferred_suffixes=definition["preferred_suffixes"],
                excluded_suffixes=definition["excluded_suffixes"]
            )

        # Default: UNKNOWN - nema filtriranja
        return QueryTypeResult(
            query_type=QueryType.UNKNOWN,
            confidence=0.0,
            matched_pattern="",
            preferred_suffixes=[],
            excluded_suffixes=[]
        )

    def filter_tools_by_type(
        self,
        tools: List[str],
        query_type_result: QueryTypeResult,
        min_confidence: float = 0.7
    ) -> List[str]:
        """
        Filtrira alate prema tipu upita.

        Args:
            tools: Lista tool_id-eva
            query_type_result: Rezultat klasifikacije
            min_confidence: Minimalna confidence za primjenu filtera

        Returns:
            Filtrirana lista alata
        """
        # Ako confidence nije dovoljan, ne filtriraj
        if query_type_result.confidence < min_confidence:
            return tools

        # Ako nema pravila, ne filtriraj
        if not query_type_result.preferred_suffixes and not query_type_result.excluded_suffixes:
            return tools

        preferred = query_type_result.preferred_suffixes
        excluded = query_type_result.excluded_suffixes

        filtered = []
        preferred_matches = []

        for tool_id in tools:
            tool_lower = tool_id.lower()

            # Provjeri excluded suffixes
            is_excluded = any(
                tool_lower.endswith(suffix.lower())
                for suffix in excluded
            )

            if is_excluded:
                continue

            # Provjeri preferred suffixes
            is_preferred = any(
                tool_lower.endswith(suffix.lower())
                for suffix in preferred
            )

            if is_preferred:
                preferred_matches.append(tool_id)
            else:
                filtered.append(tool_id)

        # Preferred matches idu na početak
        result = preferred_matches + filtered

        logger.debug(
            f"QueryTypeClassifier filter: {len(tools)} -> {len(result)} tools "
            f"(preferred: {len(preferred_matches)}, type: {query_type_result.query_type.value})"
        )

        return result

    def boost_preferred_tools(
        self,
        tool_scores: List[Tuple[str, float]],
        query_type_result: QueryTypeResult,
        boost_factor: float = 1.25
    ) -> List[Tuple[str, float]]:
        """
        Boost score za alate koji odgovaraju tipu upita.

        Args:
            tool_scores: Lista (tool_id, score) parova
            query_type_result: Rezultat klasifikacije
            boost_factor: Faktor boosta (default 1.25 = 25%)

        Returns:
            Lista s boostanim scoreovima
        """
        if query_type_result.confidence < 0.7:
            return tool_scores

        preferred = query_type_result.preferred_suffixes

        if not preferred:
            return tool_scores

        boosted = []
        for tool_id, score in tool_scores:
            tool_lower = tool_id.lower()

            is_preferred = any(
                tool_lower.endswith(suffix.lower())
                for suffix in preferred
            )

            if is_preferred:
                new_score = min(score * boost_factor, 1.0)
                boosted.append((tool_id, new_score))
            else:
                boosted.append((tool_id, score))

        # Resortiraj po novom scoreu
        boosted.sort(key=lambda x: x[1], reverse=True)

        return boosted


# Singleton instance
_classifier: Optional[QueryTypeClassifier] = None


def get_query_type_classifier() -> QueryTypeClassifier:
    """Dohvati singleton instancu classifiera."""
    global _classifier
    if _classifier is None:
        _classifier = QueryTypeClassifier()
    return _classifier


def classify_query_type(query: str) -> QueryTypeResult:
    """
    Convenience funkcija za klasifikaciju upita.

    Args:
        query: Korisnički upit

    Returns:
        QueryTypeResult
    """
    classifier = get_query_type_classifier()
    return classifier.classify(query)
