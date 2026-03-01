"""
Embedding Evaluator - Measures actual search accuracy with MRR and NDCG metrics.

Industry-standard metrics for Information Retrieval:
- MRR (Mean Reciprocal Rank): How highly is the correct result ranked?
- NDCG (Normalized Discounted Cumulative Gain): Quality of ranking with relevance scores
- Hit@K: Is the correct result in top K?

This addresses the critical problem: "Accuracy - UNKNOWN" → "Know baseline"
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class QueryTestCase:
    """A single test case for evaluation."""

    query: str                          # User's Croatian query
    expected_tool_id: str               # Correct tool operation_id
    relevance_scores: Dict[str, int] = field(default_factory=dict)
    # Optionally: {"tool_id": relevance_score}
    # relevance 3 = perfect match, 2 = acceptable, 1 = partial, 0 = irrelevant
    category: str = "general"           # For grouping (vehicle, booking, etc.)


@dataclass
class EvaluationResult:
    """Results from running evaluation."""

    # Core metrics
    mrr: float = 0.0                    # Mean Reciprocal Rank (0-1, higher is better)
    ndcg_at_5: float = 0.0              # NDCG@5 (0-1, higher is better)
    ndcg_at_10: float = 0.0             # NDCG@10 (0-1, higher is better)
    hit_at_1: float = 0.0               # Precision@1 (0-1)
    hit_at_3: float = 0.0               # Hit rate in top 3 (0-1)
    hit_at_5: float = 0.0               # Hit rate in top 5 (0-1)
    hit_at_10: float = 0.0              # Hit rate in top 10 (0-1)

    # Details
    total_queries: int = 0
    successful_queries: int = 0         # Found in top 10
    failed_queries: List[str] = field(default_factory=list)

    # Per-category breakdown
    category_mrr: Dict[str, float] = field(default_factory=dict)

    # Timestamp
    evaluated_at: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "metrics": {
                "mrr": round(self.mrr, 4),
                "ndcg@5": round(self.ndcg_at_5, 4),
                "ndcg@10": round(self.ndcg_at_10, 4),
                "hit@1": round(self.hit_at_1, 4),
                "hit@3": round(self.hit_at_3, 4),
                "hit@5": round(self.hit_at_5, 4),
                "hit@10": round(self.hit_at_10, 4),
            },
            "summary": {
                "total_queries": self.total_queries,
                "successful_queries": self.successful_queries,
                "failed_queries_count": len(self.failed_queries),
                "failed_queries_sample": self.failed_queries[:10],
            },
            "category_breakdown": self.category_mrr,
            "quality_grade": self._calculate_grade(),
            "evaluated_at": self.evaluated_at,
        }

    def _calculate_grade(self) -> str:
        """Calculate quality grade based on MRR."""
        if self.mrr >= 0.9:
            return "A+ (Excellent)"
        elif self.mrr >= 0.8:
            return "A (Very Good)"
        elif self.mrr >= 0.7:
            return "B (Good)"
        elif self.mrr >= 0.6:
            return "C (Acceptable)"
        elif self.mrr >= 0.5:
            return "D (Poor)"
        else:
            return "F (Critical)"


class EmbeddingEvaluator:
    """
    Evaluates embedding/search quality using industry-standard IR metrics.

    Usage:
        evaluator = EmbeddingEvaluator()
        test_cases = evaluator.load_test_set("evaluation_queries.json")
        results = evaluator.evaluate(search_function, test_cases)
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize evaluator."""
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)

    def create_initial_test_set(self) -> List[QueryTestCase]:
        """
        Create comprehensive evaluation dataset with 140 queries.

        Categories:
        - vehicle: Vehicle-related queries (most common)
        - booking: Reservation/booking queries
        - maintenance: Service/repair queries
        - financial: Cost/payment queries
        - location: Location-based queries
        - person: Driver/user queries
        - schedule: Calendar/availability queries
        - document: Documents and reports
        - fleet: Fleet management
        - fuel: Fuel-related queries
        """
        test_cases = [
            # ---
            # VEHICLE QUERIES (30 queries - most common use case)
            # ---
            QueryTestCase(query="prikaži mi vozilo", expected_tool_id="get_Vehicle", category="vehicle"),
            QueryTestCase(query="daj mi podatke o autu", expected_tool_id="get_Vehicle", category="vehicle"),
            QueryTestCase(query="informacije o vozilu", expected_tool_id="get_Vehicle", category="vehicle"),
            QueryTestCase(query="detalji auta", expected_tool_id="get_Vehicle", category="vehicle"),
            QueryTestCase(query="pokaži auto", expected_tool_id="get_Vehicle", category="vehicle"),
            QueryTestCase(query="dohvati vozilo", expected_tool_id="get_Vehicle", category="vehicle"),
            QueryTestCase(query="kilometraža vozila", expected_tool_id="get_VehicleMileage", category="vehicle"),
            QueryTestCase(query="koliko je prešao auto", expected_tool_id="get_VehicleMileage", category="vehicle"),
            QueryTestCase(query="prijeđeni kilometri", expected_tool_id="get_VehicleMileage", category="vehicle"),
            QueryTestCase(query="km na autu", expected_tool_id="get_VehicleMileage", category="vehicle"),
            QueryTestCase(query="stanje kilometara", expected_tool_id="get_VehicleMileage", category="vehicle"),
            QueryTestCase(query="odometar", expected_tool_id="get_VehicleMileage", category="vehicle"),
            QueryTestCase(query="pokaži sve aute", expected_tool_id="get_Vehicles", category="vehicle"),
            QueryTestCase(query="lista vozila", expected_tool_id="get_Vehicles", category="vehicle"),
            QueryTestCase(query="sva vozila", expected_tool_id="get_Vehicles", category="vehicle"),
            QueryTestCase(query="popis automobila", expected_tool_id="get_Vehicles", category="vehicle"),
            QueryTestCase(query="vozni park", expected_tool_id="get_Vehicles", category="vehicle"),
            QueryTestCase(query="daj mi listu auta", expected_tool_id="get_Vehicles", category="vehicle"),
            QueryTestCase(query="registracija vozila", expected_tool_id="get_VehicleRegistration", category="vehicle"),
            QueryTestCase(query="tablice auta", expected_tool_id="get_VehicleRegistration", category="vehicle"),
            QueryTestCase(query="reg oznaka", expected_tool_id="get_VehicleRegistration", category="vehicle"),
            QueryTestCase(query="registarske tablice", expected_tool_id="get_VehicleRegistration", category="vehicle"),
            QueryTestCase(query="dodaj novo vozilo", expected_tool_id="post_Vehicle", category="vehicle"),
            QueryTestCase(query="kreiraj auto", expected_tool_id="post_Vehicle", category="vehicle"),
            QueryTestCase(query="unesi novo vozilo", expected_tool_id="post_Vehicle", category="vehicle"),
            QueryTestCase(query="ažuriraj vozilo", expected_tool_id="put_Vehicle", category="vehicle"),
            QueryTestCase(query="promijeni podatke auta", expected_tool_id="put_Vehicle", category="vehicle"),
            QueryTestCase(query="obriši vozilo", expected_tool_id="delete_Vehicle", category="vehicle"),
            QueryTestCase(query="ukloni auto", expected_tool_id="delete_Vehicle", category="vehicle"),
            QueryTestCase(query="status vozila", expected_tool_id="get_VehicleStatus", category="vehicle"),

            # ---
            # BOOKING QUERIES (20 queries)
            # ---
            QueryTestCase(query="rezerviraj auto", expected_tool_id="post_Booking", category="booking"),
            QueryTestCase(query="napravi rezervaciju", expected_tool_id="post_Booking", category="booking"),
            QueryTestCase(query="zakaži najam", expected_tool_id="post_Booking", category="booking"),
            QueryTestCase(query="kreiraj booking", expected_tool_id="post_Booking", category="booking"),
            QueryTestCase(query="nova rezervacija", expected_tool_id="post_Booking", category="booking"),
            QueryTestCase(query="rezerviraj vozilo za sutra", expected_tool_id="post_Booking", category="booking"),
            QueryTestCase(query="prikaži rezervacije", expected_tool_id="get_Bookings", category="booking"),
            QueryTestCase(query="moje rezervacije", expected_tool_id="get_Bookings", category="booking"),
            QueryTestCase(query="lista bookinga", expected_tool_id="get_Bookings", category="booking"),
            QueryTestCase(query="sve rezervacije", expected_tool_id="get_Bookings", category="booking"),
            QueryTestCase(query="aktivne rezervacije", expected_tool_id="get_Bookings", category="booking"),
            QueryTestCase(query="otkaži rezervaciju", expected_tool_id="delete_Booking", category="booking"),
            QueryTestCase(query="poništi booking", expected_tool_id="delete_Booking", category="booking"),
            QueryTestCase(query="obriši rezervaciju", expected_tool_id="delete_Booking", category="booking"),
            QueryTestCase(query="promijeni rezervaciju", expected_tool_id="put_Booking", category="booking"),
            QueryTestCase(query="ažuriraj booking", expected_tool_id="put_Booking", category="booking"),
            QueryTestCase(query="izmjeni rezervaciju", expected_tool_id="put_Booking", category="booking"),
            QueryTestCase(query="detalji rezervacije", expected_tool_id="get_Booking", category="booking"),
            QueryTestCase(query="prikaži booking", expected_tool_id="get_Booking", category="booking"),
            QueryTestCase(query="kalendar rezervacija", expected_tool_id="get_BookingCalendar", category="booking"),

            # ---
            # MAINTENANCE QUERIES (15 queries)
            # ---
            QueryTestCase(query="prijavi kvar", expected_tool_id="post_Damage", category="maintenance"),
            QueryTestCase(query="unesi štetu", expected_tool_id="post_Damage", category="maintenance"),
            QueryTestCase(query="nova šteta", expected_tool_id="post_Damage", category="maintenance"),
            QueryTestCase(query="prijavi oštećenje", expected_tool_id="post_Damage", category="maintenance"),
            QueryTestCase(query="popis šteta", expected_tool_id="get_Damages", category="maintenance"),
            QueryTestCase(query="sve štete", expected_tool_id="get_Damages", category="maintenance"),
            QueryTestCase(query="lista oštećenja", expected_tool_id="get_Damages", category="maintenance"),
            QueryTestCase(query="servis vozila", expected_tool_id="get_VehicleMaintenance", category="maintenance"),
            QueryTestCase(query="održavanje auta", expected_tool_id="get_VehicleMaintenance", category="maintenance"),
            QueryTestCase(query="povijest servisa", expected_tool_id="get_VehicleMaintenance", category="maintenance"),
            QueryTestCase(query="zakaži servis", expected_tool_id="post_MaintenanceAppointment", category="maintenance"),
            QueryTestCase(query="popravak vozila", expected_tool_id="get_VehicleRepairs", category="maintenance"),
            QueryTestCase(query="inspekcija auta", expected_tool_id="get_VehicleInspection", category="maintenance"),
            QueryTestCase(query="tehnički pregled", expected_tool_id="get_VehicleInspection", category="maintenance"),
            QueryTestCase(query="status servisa", expected_tool_id="get_MaintenanceStatus", category="maintenance"),

            # ---
            # FINANCIAL QUERIES (15 queries)
            # ---
            QueryTestCase(query="cijena najma", expected_tool_id="get_BookingPrice", category="financial"),
            QueryTestCase(query="izračunaj cijenu", expected_tool_id="get_BookingPrice", category="financial"),
            QueryTestCase(query="troškovi vozila", expected_tool_id="get_VehicleCosts", category="financial"),
            QueryTestCase(query="koliko košta", expected_tool_id="get_BookingPrice", category="financial"),
            QueryTestCase(query="računi", expected_tool_id="get_Invoices", category="financial"),
            QueryTestCase(query="fakture", expected_tool_id="get_Invoices", category="financial"),
            QueryTestCase(query="lista računa", expected_tool_id="get_Invoices", category="financial"),
            QueryTestCase(query="napravi račun", expected_tool_id="post_Invoice", category="financial"),
            QueryTestCase(query="kreiraj fakturu", expected_tool_id="post_Invoice", category="financial"),
            QueryTestCase(query="plaćanja", expected_tool_id="get_Payments", category="financial"),
            QueryTestCase(query="transakcije", expected_tool_id="get_Transactions", category="financial"),
            QueryTestCase(query="ukupni troškovi", expected_tool_id="get_CostsSummary", category="financial"),
            QueryTestCase(query="financijski izvještaj", expected_tool_id="get_FinancialReport", category="financial"),
            QueryTestCase(query="stanje računa", expected_tool_id="get_AccountBalance", category="financial"),
            QueryTestCase(query="cjenik", expected_tool_id="get_Pricing", category="financial"),

            # ---
            # LOCATION QUERIES (10 queries)
            # ---
            QueryTestCase(query="lokacija vozila", expected_tool_id="get_VehicleLocation", category="location"),
            QueryTestCase(query="gdje je auto", expected_tool_id="get_VehicleLocation", category="location"),
            QueryTestCase(query="GPS pozicija", expected_tool_id="get_VehicleLocation", category="location"),
            QueryTestCase(query="prati vozilo", expected_tool_id="get_VehicleTracking", category="location"),
            QueryTestCase(query="popis lokacija", expected_tool_id="get_Locations", category="location"),
            QueryTestCase(query="sve poslovnice", expected_tool_id="get_Branches", category="location"),
            QueryTestCase(query="najbliža lokacija", expected_tool_id="get_NearestLocation", category="location"),
            QueryTestCase(query="adrese", expected_tool_id="get_Addresses", category="location"),
            QueryTestCase(query="zona preuzimanja", expected_tool_id="get_PickupZones", category="location"),
            QueryTestCase(query="mjesto vraćanja", expected_tool_id="get_ReturnLocations", category="location"),

            # ---
            # PERSON/DRIVER QUERIES (15 queries)
            # ---
            QueryTestCase(query="podaci o vozaču", expected_tool_id="get_Driver", category="person"),
            QueryTestCase(query="informacije o vozaču", expected_tool_id="get_Driver", category="person"),
            QueryTestCase(query="lista vozača", expected_tool_id="get_Drivers", category="person"),
            QueryTestCase(query="svi vozači", expected_tool_id="get_Drivers", category="person"),
            QueryTestCase(query="dodaj novog vozača", expected_tool_id="post_Driver", category="person"),
            QueryTestCase(query="registriraj vozača", expected_tool_id="post_Driver", category="person"),
            QueryTestCase(query="ažuriraj vozača", expected_tool_id="put_Driver", category="person"),
            QueryTestCase(query="obriši vozača", expected_tool_id="delete_Driver", category="person"),
            QueryTestCase(query="korisnici", expected_tool_id="get_Users", category="person"),
            QueryTestCase(query="lista korisnika", expected_tool_id="get_Users", category="person"),
            QueryTestCase(query="dodaj korisnika", expected_tool_id="post_User", category="person"),
            QueryTestCase(query="kontakti", expected_tool_id="get_Contacts", category="person"),
            QueryTestCase(query="kupci", expected_tool_id="get_Customers", category="person"),
            QueryTestCase(query="zaposlenici", expected_tool_id="get_Employees", category="person"),
            QueryTestCase(query="vozačka dozvola", expected_tool_id="get_DriverLicense", category="person"),

            # ---
            # SCHEDULE/AVAILABILITY QUERIES (10 queries)
            # ---
            QueryTestCase(query="raspored", expected_tool_id="get_Schedule", category="schedule"),
            QueryTestCase(query="kalendar", expected_tool_id="get_Calendar", category="schedule"),
            QueryTestCase(query="dostupnost vozila", expected_tool_id="get_VehicleAvailability", category="schedule"),
            QueryTestCase(query="slobodna vozila", expected_tool_id="get_AvailableVehicles", category="schedule"),
            QueryTestCase(query="termini", expected_tool_id="get_TimeSlots", category="schedule"),
            QueryTestCase(query="slobodni termini", expected_tool_id="get_AvailableSlots", category="schedule"),
            QueryTestCase(query="smjene", expected_tool_id="get_Shifts", category="schedule"),
            QueryTestCase(query="radni sati", expected_tool_id="get_WorkingHours", category="schedule"),
            QueryTestCase(query="plan rada", expected_tool_id="get_WorkPlan", category="schedule"),
            QueryTestCase(query="zauzetost", expected_tool_id="get_Occupancy", category="schedule"),

            # ---
            # FUEL QUERIES (10 queries)
            # ---
            QueryTestCase(query="stanje goriva", expected_tool_id="get_VehicleFuelLevel", category="fuel"),
            QueryTestCase(query="razina goriva", expected_tool_id="get_VehicleFuelLevel", category="fuel"),
            QueryTestCase(query="koliko ima goriva", expected_tool_id="get_VehicleFuelLevel", category="fuel"),
            QueryTestCase(query="tank", expected_tool_id="get_VehicleFuelLevel", category="fuel"),
            QueryTestCase(query="potrošnja goriva", expected_tool_id="get_FuelConsumption", category="fuel"),
            QueryTestCase(query="unesi gorivo", expected_tool_id="post_FuelEntry", category="fuel"),
            QueryTestCase(query="prijavi točenje", expected_tool_id="post_FuelEntry", category="fuel"),
            QueryTestCase(query="povijest točenja", expected_tool_id="get_FuelHistory", category="fuel"),
            QueryTestCase(query="kartice za gorivo", expected_tool_id="get_FuelCards", category="fuel"),
            QueryTestCase(query="troškovi goriva", expected_tool_id="get_FuelCosts", category="fuel"),

            # ---
            # DOCUMENT QUERIES (10 queries)
            # ---
            QueryTestCase(query="dokumenti", expected_tool_id="get_Documents", category="document"),
            QueryTestCase(query="ugovori", expected_tool_id="get_Contracts", category="document"),
            QueryTestCase(query="police osiguranja", expected_tool_id="get_InsurancePolicies", category="document"),
            QueryTestCase(query="izvještaji", expected_tool_id="get_Reports", category="document"),
            QueryTestCase(query="generiraj izvještaj", expected_tool_id="post_Report", category="document"),
            QueryTestCase(query="certifikati", expected_tool_id="get_Certificates", category="document"),
            QueryTestCase(query="licence", expected_tool_id="get_Licenses", category="document"),
            QueryTestCase(query="bilješke", expected_tool_id="get_Notes", category="document"),
            QueryTestCase(query="dodaj bilješku", expected_tool_id="post_Note", category="document"),
            QueryTestCase(query="privici", expected_tool_id="get_Attachments", category="document"),

            # ---
            # FLEET MANAGEMENT (15 queries)
            # ---
            QueryTestCase(query="statistika flote", expected_tool_id="get_FleetStatistics", category="fleet"),
            QueryTestCase(query="pregled voznog parka", expected_tool_id="get_FleetOverview", category="fleet"),
            QueryTestCase(query="iskorištenost flote", expected_tool_id="get_FleetUtilization", category="fleet"),
            QueryTestCase(query="kategorije vozila", expected_tool_id="get_VehicleCategories", category="fleet"),
            QueryTestCase(query="tipovi vozila", expected_tool_id="get_VehicleTypes", category="fleet"),
            QueryTestCase(query="klase automobila", expected_tool_id="get_VehicleClasses", category="fleet"),
            QueryTestCase(query="modeli vozila", expected_tool_id="get_VehicleModels", category="fleet"),
            QueryTestCase(query="marke automobila", expected_tool_id="get_VehicleBrands", category="fleet"),
            QueryTestCase(query="specifikacije vozila", expected_tool_id="get_VehicleSpecifications", category="fleet"),
            QueryTestCase(query="oprema vozila", expected_tool_id="get_VehicleEquipment", category="fleet"),
            QueryTestCase(query="dodaci vozila", expected_tool_id="get_VehicleAccessories", category="fleet"),
            QueryTestCase(query="SUV vozila", expected_tool_id="get_SUVs", category="fleet"),
            QueryTestCase(query="električna vozila", expected_tool_id="get_ElectricVehicles", category="fleet"),
            QueryTestCase(query="kombiji", expected_tool_id="get_Vans", category="fleet"),
            QueryTestCase(query="luksuzna vozila", expected_tool_id="get_LuxuryVehicles", category="fleet"),

            # ---
            # DAMAGE/ACCIDENT QUERIES (15 queries)
            # ---
            QueryTestCase(query="prijavi štetu", expected_tool_id="post_DamageReport", category="damage"),
            QueryTestCase(query="oštećenja vozila", expected_tool_id="get_VehicleDamages", category="damage"),
            QueryTestCase(query="štete", expected_tool_id="get_Damages", category="damage"),
            QueryTestCase(query="nesreće", expected_tool_id="get_Accidents", category="damage"),
            QueryTestCase(query="prijava nesreće", expected_tool_id="post_AccidentReport", category="damage"),
            QueryTestCase(query="procjena štete", expected_tool_id="get_DamageAssessment", category="damage"),
            QueryTestCase(query="fotografije štete", expected_tool_id="get_DamagePhotos", category="damage"),
            QueryTestCase(query="status štete", expected_tool_id="get_DamageStatus", category="damage"),
            QueryTestCase(query="osiguranje za štetu", expected_tool_id="get_DamageInsurance", category="damage"),
            QueryTestCase(query="zahtjev za osiguranje", expected_tool_id="post_InsuranceClaim", category="damage"),
            QueryTestCase(query="popravak štete", expected_tool_id="get_DamageRepairs", category="damage"),
            QueryTestCase(query="trošak popravka", expected_tool_id="get_RepairCost", category="damage"),
            QueryTestCase(query="udar na vozilo", expected_tool_id="post_DamageReport", category="damage"),
            QueryTestCase(query="ogrebotina na autu", expected_tool_id="post_DamageReport", category="damage"),
            QueryTestCase(query="kvar vozila", expected_tool_id="post_MalfunctionReport", category="damage"),

            # ---
            # RENTAL OPERATIONS (15 queries)
            # ---
            QueryTestCase(query="preuzimanje vozila", expected_tool_id="post_VehiclePickup", category="rental_ops"),
            QueryTestCase(query="vraćanje auta", expected_tool_id="post_VehicleReturn", category="rental_ops"),
            QueryTestCase(query="primopredaja vozila", expected_tool_id="post_VehicleHandover", category="rental_ops"),
            QueryTestCase(query="checkin", expected_tool_id="post_Checkin", category="rental_ops"),
            QueryTestCase(query="checkout", expected_tool_id="post_Checkout", category="rental_ops"),
            QueryTestCase(query="produži najam", expected_tool_id="post_ExtendRental", category="rental_ops"),
            QueryTestCase(query="otkaži rezervaciju", expected_tool_id="post_CancelBooking", category="rental_ops"),
            QueryTestCase(query="promijeni rezervaciju", expected_tool_id="put_ModifyBooking", category="rental_ops"),
            QueryTestCase(query="zamjena vozila", expected_tool_id="post_VehicleSwap", category="rental_ops"),
            QueryTestCase(query="upgrade vozila", expected_tool_id="post_VehicleUpgrade", category="rental_ops"),
            QueryTestCase(query="downgrade vozila", expected_tool_id="post_VehicleDowngrade", category="rental_ops"),
            QueryTestCase(query="dodatna oprema", expected_tool_id="post_AddExtras", category="rental_ops"),
            QueryTestCase(query="naruči GPS", expected_tool_id="post_AddGPS", category="rental_ops"),
            QueryTestCase(query="dječje sjedalo", expected_tool_id="post_AddChildSeat", category="rental_ops"),
            QueryTestCase(query="dodaj dodatnog vozača", expected_tool_id="post_AddDriver", category="rental_ops"),

            # ---
            # LOYALTY/REWARDS (10 queries)
            # ---
            QueryTestCase(query="bodovi lojalnosti", expected_tool_id="get_LoyaltyPoints", category="loyalty"),
            QueryTestCase(query="nagrade", expected_tool_id="get_Rewards", category="loyalty"),
            QueryTestCase(query="kuponi", expected_tool_id="get_Coupons", category="loyalty"),
            QueryTestCase(query="vaučeri", expected_tool_id="get_Vouchers", category="loyalty"),
            QueryTestCase(query="popusti", expected_tool_id="get_Discounts", category="loyalty"),
            QueryTestCase(query="promocije", expected_tool_id="get_Promotions", category="loyalty"),
            QueryTestCase(query="članstvo", expected_tool_id="get_Membership", category="loyalty"),
            QueryTestCase(query="razina članstva", expected_tool_id="get_MembershipTier", category="loyalty"),
            QueryTestCase(query="iskoristi kupon", expected_tool_id="post_RedeemCoupon", category="loyalty"),
            QueryTestCase(query="iskoristi bodove", expected_tool_id="post_RedeemPoints", category="loyalty"),

            # ---
            # NOTIFICATIONS/ALERTS (10 queries)
            # ---
            QueryTestCase(query="obavijesti", expected_tool_id="get_Notifications", category="notifications"),
            QueryTestCase(query="upozorenja", expected_tool_id="get_Alerts", category="notifications"),
            QueryTestCase(query="poruke", expected_tool_id="get_Messages", category="notifications"),
            QueryTestCase(query="nepročitane poruke", expected_tool_id="get_UnreadMessages", category="notifications"),
            QueryTestCase(query="pošalji obavijest", expected_tool_id="post_Notification", category="notifications"),
            QueryTestCase(query="pošalji SMS", expected_tool_id="post_SMS", category="notifications"),
            QueryTestCase(query="pošalji email", expected_tool_id="post_Email", category="notifications"),
            QueryTestCase(query="podsjetnici", expected_tool_id="get_Reminders", category="notifications"),
            QueryTestCase(query="postavi podsjetnik", expected_tool_id="post_Reminder", category="notifications"),
            QueryTestCase(query="alarmi vozila", expected_tool_id="get_VehicleAlerts", category="notifications"),
        ]

        return test_cases

    def save_test_set(self, test_cases: List[QueryTestCase], filename: str = "evaluation_queries.json"):
        """Save test set to JSON file."""
        filepath = self.data_dir / filename
        data = {
            "created_at": datetime.now().isoformat(),
            "total_queries": len(test_cases),
            "test_cases": [
                {
                    "query": tc.query,
                    "expected_tool_id": tc.expected_tool_id,
                    "relevance_scores": tc.relevance_scores,
                    "category": tc.category,
                }
                for tc in test_cases
            ]
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(test_cases)} test cases to {filepath}")
        return filepath

    def load_test_set(self, filename: str = "evaluation_queries.json") -> List[QueryTestCase]:
        """Load test set from JSON file."""
        filepath = self.data_dir / filename

        if not filepath.exists():
            logger.warning(f"Test set not found: {filepath}")
            return []

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        test_cases = [
            QueryTestCase(
                query=tc["query"],
                expected_tool_id=tc["expected_tool_id"],
                relevance_scores=tc.get("relevance_scores", {}),
                category=tc.get("category", "general"),
            )
            for tc in data["test_cases"]
        ]

        logger.info(f"Loaded {len(test_cases)} test cases")
        return test_cases

    def evaluate(
        self,
        search_func,
        test_cases: List[QueryTestCase],
        top_k: int = 10
    ) -> EvaluationResult:
        """
        Evaluate search quality against test cases.

        Args:
            search_func: Function that takes query string and returns
                        list of (tool_id, score) tuples ordered by relevance
            test_cases: List of QueryTestCase to evaluate
            top_k: How many results to consider (default 10)

        Returns:
            EvaluationResult with all metrics
        """
        result = EvaluationResult(
            total_queries=len(test_cases),
            evaluated_at=datetime.now().isoformat()
        )

        if not test_cases:
            return result

        reciprocal_ranks = []
        ndcg_5_scores = []
        ndcg_10_scores = []
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_5 = 0
        hits_at_10 = 0

        # Per-category tracking
        category_rrs: Dict[str, List[float]] = {}

        for tc in test_cases:
            try:
                # Get search results
                results = search_func(tc.query)
                result_ids = [r[0] if isinstance(r, tuple) else r for r in results[:top_k]]

                # Find rank of expected tool
                try:
                    rank = result_ids.index(tc.expected_tool_id) + 1  # 1-indexed
                except ValueError:
                    rank = 0  # Not found

                # Calculate Reciprocal Rank
                if rank > 0:
                    rr = 1.0 / rank
                    result.successful_queries += 1

                    if rank == 1:
                        hits_at_1 += 1
                    if rank <= 3:
                        hits_at_3 += 1
                    if rank <= 5:
                        hits_at_5 += 1
                    if rank <= 10:
                        hits_at_10 += 1
                else:
                    rr = 0.0
                    result.failed_queries.append(tc.query)

                reciprocal_ranks.append(rr)

                # Track per category
                if tc.category not in category_rrs:
                    category_rrs[tc.category] = []
                category_rrs[tc.category].append(rr)

                # Calculate NDCG
                ndcg_5 = self._calculate_ndcg(result_ids[:5], tc, 5)
                ndcg_10 = self._calculate_ndcg(result_ids[:10], tc, 10)
                ndcg_5_scores.append(ndcg_5)
                ndcg_10_scores.append(ndcg_10)

            except Exception as e:
                logger.error(f"Error evaluating query '{tc.query}': {e}")
                reciprocal_ranks.append(0.0)
                ndcg_5_scores.append(0.0)
                ndcg_10_scores.append(0.0)
                result.failed_queries.append(f"{tc.query} (error: {e})")

        # Calculate final metrics
        result.mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        result.ndcg_at_5 = sum(ndcg_5_scores) / len(ndcg_5_scores) if ndcg_5_scores else 0.0
        result.ndcg_at_10 = sum(ndcg_10_scores) / len(ndcg_10_scores) if ndcg_10_scores else 0.0

        result.hit_at_1 = hits_at_1 / len(test_cases)
        result.hit_at_3 = hits_at_3 / len(test_cases)
        result.hit_at_5 = hits_at_5 / len(test_cases)
        result.hit_at_10 = hits_at_10 / len(test_cases)

        # Per-category MRR
        for category, rrs in category_rrs.items():
            result.category_mrr[category] = sum(rrs) / len(rrs) if rrs else 0.0

        return result

    def _calculate_ndcg(
        self,
        result_ids: List[str],
        test_case: QueryTestCase,
        k: int
    ) -> float:
        """
        Calculate NDCG@k for a single query.

        If no explicit relevance scores provided, uses binary relevance:
        - 1 if it's the expected tool
        - 0 otherwise
        """
        # Get relevance for each result
        if test_case.relevance_scores:
            relevances = [
                test_case.relevance_scores.get(r_id, 0)
                for r_id in result_ids
            ]
        else:
            # Binary relevance
            relevances = [
                1 if r_id == test_case.expected_tool_id else 0
                for r_id in result_ids
            ]

        # DCG
        dcg = 0.0
        for i, rel in enumerate(relevances[:k]):
            dcg += rel / math.log2(i + 2)  # +2 because position is 1-indexed

        # Ideal DCG (best possible ranking)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_relevances[:k]):
            idcg += rel / math.log2(i + 2)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def print_report(self, result: EvaluationResult) -> None:
        """Print human-readable evaluation report."""
        print("\n" + "=" * 60)
        print("EMBEDDING SEARCH EVALUATION REPORT")
        print("=" * 60)

        print(f"\n📊 CORE METRICS:")
        print(f"   MRR (Mean Reciprocal Rank): {result.mrr:.4f}")
        print(f"   NDCG@5:  {result.ndcg_at_5:.4f}")
        print(f"   NDCG@10: {result.ndcg_at_10:.4f}")

        print(f"\n🎯 HIT RATES:")
        print(f"   Hit@1:  {result.hit_at_1 * 100:.1f}%")
        print(f"   Hit@3:  {result.hit_at_3 * 100:.1f}%")
        print(f"   Hit@5:  {result.hit_at_5 * 100:.1f}%")
        print(f"   Hit@10: {result.hit_at_10 * 100:.1f}%")

        print(f"\n📈 SUMMARY:")
        print(f"   Total queries: {result.total_queries}")
        print(f"   Successful: {result.successful_queries}")
        print(f"   Failed: {len(result.failed_queries)}")

        if result.category_mrr:
            print(f"\n📁 BY CATEGORY:")
            for cat, mrr in sorted(result.category_mrr.items()):
                print(f"   {cat}: MRR={mrr:.4f}")

        grade = result.to_dict()["quality_grade"]
        print(f"\n🏆 QUALITY GRADE: {grade}")

        if result.failed_queries:
            print(f"\n❌ FAILED QUERIES (sample):")
            for q in result.failed_queries[:5]:
                print(f"   - {q}")

        print("=" * 60 + "\n")


# Convenience function
def create_evaluation_dataset():
    """Create and save initial evaluation dataset."""
    evaluator = EmbeddingEvaluator()
    test_cases = evaluator.create_initial_test_set()
    filepath = evaluator.save_test_set(test_cases)
    print(f"Created evaluation dataset with {len(test_cases)} queries: {filepath}")
    return test_cases
