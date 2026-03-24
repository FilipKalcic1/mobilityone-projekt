"""
Fix Generic Tool Documentation

This script updates tool_documentation.json to add entity-specific
keywords and example queries for all generic tools (_Agg, _GroupBy, etc.)

Fixes ~300 tools that currently have generic documentation.
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# Entity mappings - Croatian names and synonyms for each entity type
ENTITY_MAPPINGS = {
    "Vehicles": {
        "hr_name": "vozila",
        "hr_singular": "vozilo",
        "hr_genitive": "vozila",
        "synonyms": ["vozilo", "vozila", "auto", "automobil", "flota", "registracija", "tablica"],
        "keywords": ["kilometraža", "registracija", "marka", "model", "gorivo", "servis"],
        "examples": {
            "_Agg": [
                "Kolika je prosječna kilometraža vozila?",
                "Daj mi statistiku vozila u floti",
                "Prosječna starost vozila",
                "Maksimalna kilometraža u floti"
            ],
            "_GroupBy": [
                "Grupiraj vozila po marki",
                "Vozila grupirana po tipu goriva",
                "Grupiraj flotu po godini proizvodnje",
                "Vozila po kategorijama"
            ],
            "_ProjectTo": [
                "Daj mi vozila s kolonama: registracija, marka, model",
                "Lista vozila samo s osnovnim podacima",
                "Vozila - samo ID i naziv"
            ],
            "_metadata": [
                "Metapodaci za vozilo",
                "Dodatne informacije o vozilu",
                "Prošireni podaci vozila"
            ],
            "_DeleteByCriteria": [
                "Obriši sva neaktivna vozila",
                "Masovno brisanje vozila po kriteriju",
                "Ukloni vozila starija od 10 godina"
            ],
            "_multipatch": [
                "Masovno ažuriraj vozila",
                "Ažuriraj više vozila odjednom",
                "Bulk update flote vozila"
            ]
        }
    },
    "Companies": {
        "hr_name": "kompanije",
        "hr_singular": "kompanija",
        "hr_genitive": "kompanija",
        "synonyms": ["kompanija", "tvrtka", "firma", "poduzeće", "organizacija"],
        "keywords": ["naziv", "OIB", "adresa", "vlasnik"],
        "examples": {
            "_Agg": [
                "Statistika po kompanijama",
                "Prosječan broj vozila po kompaniji",
                "Agregirani podaci kompanija"
            ],
            "_GroupBy": [
                "Grupiraj kompanije po regiji",
                "Kompanije grupirane po veličini flote",
                "Grupiraj tvrtke po broju zaposlenika"
            ],
            "_ProjectTo": [
                "Daj mi kompanije s kolonama: naziv, OIB",
                "Lista kompanija samo s osnovnim podacima"
            ],
            "_metadata": [
                "Metapodaci za kompaniju",
                "Dodatne informacije o tvrtki"
            ],
            "_DeleteByCriteria": [
                "Obriši neaktivne kompanije",
                "Masovno brisanje kompanija po kriteriju"
            ],
            "_multipatch": [
                "Masovno ažuriraj kompanije",
                "Ažuriraj više kompanija odjednom"
            ]
        }
    },
    "Persons": {
        "hr_name": "osobe",
        "hr_singular": "osoba",
        "hr_genitive": "osoba",
        "synonyms": ["osoba", "korisnik", "zaposlenik", "vozač", "djelatnik"],
        "keywords": ["ime", "prezime", "email", "telefon", "OIB"],
        "examples": {
            "_Agg": [
                "Statistika zaposlenika",
                "Prosječan broj vožnji po vozaču",
                "Agregirani podaci osoba"
            ],
            "_GroupBy": [
                "Grupiraj zaposlenike po odjelu",
                "Osobe grupirane po poziciji",
                "Grupiraj vozače po iskustvu"
            ],
            "_ProjectTo": [
                "Daj mi osobe s kolonama: ime, prezime, email",
                "Lista zaposlenika samo s kontaktima"
            ],
            "_metadata": [
                "Metapodaci za osobu",
                "Dodatne informacije o zaposleniku"
            ],
            "_DeleteByCriteria": [
                "Obriši neaktivne zaposlenike",
                "Masovno brisanje osoba po kriteriju"
            ],
            "_multipatch": [
                "Masovno ažuriraj osobe",
                "Ažuriraj više zaposlenika odjednom"
            ]
        }
    },
    "Expenses": {
        "hr_name": "troškovi",
        "hr_singular": "trošak",
        "hr_genitive": "troškova",
        "synonyms": ["trošak", "troškovi", "izdatak", "račun", "faktura", "cijena"],
        "keywords": ["iznos", "datum", "vrsta", "kategorija", "dobavljač"],
        "examples": {
            "_Agg": [
                "Ukupni troškovi za mjesec",
                "Prosječni troškovi po vozilu",
                "Statistika troškova flote",
                "Suma svih izdataka"
            ],
            "_GroupBy": [
                "Grupiraj troškove po mjesecu",
                "Troškovi grupirani po kategoriji",
                "Grupiraj izdatke po vozilu",
                "Troškovi po vrsti goriva"
            ],
            "_ProjectTo": [
                "Daj mi troškove s kolonama: datum, iznos, vrsta",
                "Lista troškova samo s iznosima"
            ],
            "_metadata": [
                "Metapodaci za trošak",
                "Dodatne informacije o izdatku"
            ],
            "_DeleteByCriteria": [
                "Obriši stare troškove",
                "Masovno brisanje troškova po kriteriju"
            ],
            "_multipatch": [
                "Masovno ažuriraj troškove",
                "Ažuriraj više izdataka odjednom"
            ]
        }
    },
    "Trips": {
        "hr_name": "putovanja",
        "hr_singular": "putovanje",
        "hr_genitive": "putovanja",
        "synonyms": ["putovanje", "trip", "vožnja", "ruta", "relacija"],
        "keywords": ["početak", "kraj", "udaljenost", "trajanje", "destinacija"],
        "examples": {
            "_Agg": [
                "Ukupna kilometraža putovanja",
                "Prosječno trajanje putovanja",
                "Statistika putovanja",
                "Suma prijeđenih kilometara"
            ],
            "_GroupBy": [
                "Grupiraj putovanja po vozaču",
                "Putovanja grupirana po mjesecu",
                "Grupiraj tripove po destinaciji"
            ],
            "_ProjectTo": [
                "Daj mi putovanja s kolonama: datum, km, vozač",
                "Lista putovanja samo s osnovnim podacima"
            ],
            "_metadata": [
                "Metapodaci za putovanje",
                "Dodatne informacije o tripu"
            ],
            "_DeleteByCriteria": [
                "Obriši stara putovanja",
                "Masovno brisanje tripova po kriteriju"
            ],
            "_multipatch": [
                "Masovno ažuriraj putovanja",
                "Ažuriraj više tripova odjednom"
            ]
        }
    },
    "Cases": {
        "hr_name": "slučajevi",
        "hr_singular": "slučaj",
        "hr_genitive": "slučajeva",
        "synonyms": ["slučaj", "šteta", "kvar", "incident", "prijava", "zahtjev"],
        "keywords": ["opis", "status", "prioritet", "datum", "vrsta"],
        "examples": {
            "_Agg": [
                "Statistika prijavljenih šteta",
                "Prosječno vrijeme rješavanja slučajeva",
                "Broj otvorenih incidenata"
            ],
            "_GroupBy": [
                "Grupiraj slučajeve po statusu",
                "Štete grupirane po tipu",
                "Grupiraj incidente po prioritetu"
            ],
            "_ProjectTo": [
                "Daj mi slučajeve s kolonama: datum, status, opis",
                "Lista šteta samo s osnovnim podacima"
            ],
            "_metadata": [
                "Metapodaci za slučaj",
                "Dodatne informacije o šteti"
            ],
            "_DeleteByCriteria": [
                "Obriši zatvorene slučajeve",
                "Masovno brisanje starih incidenata"
            ],
            "_multipatch": [
                "Masovno ažuriraj slučajeve",
                "Ažuriraj više šteta odjednom"
            ]
        }
    },
    "Equipment": {
        "hr_name": "oprema",
        "hr_singular": "oprema",
        "hr_genitive": "opreme",
        "synonyms": ["oprema", "uređaj", "alat", "inventar", "asset"],
        "keywords": ["naziv", "serijski broj", "lokacija", "status"],
        "examples": {
            "_Agg": [
                "Statistika opreme",
                "Prosječna starost opreme",
                "Ukupna vrijednost inventara"
            ],
            "_GroupBy": [
                "Grupiraj opremu po lokaciji",
                "Oprema grupirana po tipu",
                "Grupiraj uređaje po statusu"
            ],
            "_ProjectTo": [
                "Daj mi opremu s kolonama: naziv, lokacija, status",
                "Lista uređaja samo s osnovnim podacima"
            ],
            "_metadata": [
                "Metapodaci za opremu",
                "Dodatne informacije o uređaju"
            ],
            "_DeleteByCriteria": [
                "Obriši neaktivnu opremu",
                "Masovno brisanje opreme po kriteriju"
            ],
            "_multipatch": [
                "Masovno ažuriraj opremu",
                "Ažuriraj više uređaja odjednom"
            ]
        }
    },
    "Partners": {
        "hr_name": "partneri",
        "hr_singular": "partner",
        "hr_genitive": "partnera",
        "synonyms": ["partner", "dobavljač", "klijent", "suradnik"],
        "keywords": ["naziv", "kontakt", "adresa", "tip"],
        "examples": {
            "_Agg": [
                "Statistika partnera",
                "Prosječan promet po partneru",
                "Agregirani podaci dobavljača"
            ],
            "_GroupBy": [
                "Grupiraj partnere po tipu",
                "Partneri grupirani po regiji",
                "Grupiraj dobavljače po kategoriji"
            ],
            "_ProjectTo": [
                "Daj mi partnere s kolonama: naziv, kontakt",
                "Lista dobavljača samo s osnovnim podacima"
            ],
            "_metadata": [
                "Metapodaci za partnera",
                "Dodatne informacije o dobavljaču"
            ],
            "_DeleteByCriteria": [
                "Obriši neaktivne partnere",
                "Masovno brisanje dobavljača po kriteriju"
            ],
            "_multipatch": [
                "Masovno ažuriraj partnere",
                "Ažuriraj više dobavljača odjednom"
            ]
        }
    },
    "CostCenters": {
        "hr_name": "troškovni centri",
        "hr_singular": "troškovni centar",
        "hr_genitive": "troškovnih centara",
        "synonyms": ["troškovni centar", "cost center", "mjesto troška", "centar troška"],
        "keywords": ["naziv", "kod", "budget", "odgovorna osoba"],
        "examples": {
            "_Agg": [
                "Statistika troškovnih centara",
                "Prosječni troškovi po centru",
                "Agregirani podaci cost centara"
            ],
            "_GroupBy": [
                "Grupiraj troškovne centre po odjelu",
                "Cost centri grupirani po budgetu"
            ],
            "_ProjectTo": [
                "Daj mi troškovne centre s kolonama: naziv, kod",
                "Lista cost centara samo s osnovnim podacima"
            ],
            "_metadata": [
                "Metapodaci za troškovni centar",
                "Dodatne informacije o cost centru"
            ],
            "_DeleteByCriteria": [
                "Obriši neaktivne troškovne centre",
                "Masovno brisanje cost centara"
            ],
            "_multipatch": [
                "Masovno ažuriraj troškovne centre",
                "Ažuriraj više cost centara odjednom"
            ]
        }
    },
    "OrgUnits": {
        "hr_name": "organizacijske jedinice",
        "hr_singular": "organizacijska jedinica",
        "hr_genitive": "organizacijskih jedinica",
        "synonyms": ["org jedinica", "odjel", "sektor", "poslovnica", "organizacijska jedinica"],
        "keywords": ["naziv", "šifra", "nadređena jedinica", "lokacija"],
        "examples": {
            "_Agg": [
                "Statistika org jedinica",
                "Prosječna veličina odjela",
                "Agregirani podaci organizacijskih jedinica"
            ],
            "_GroupBy": [
                "Grupiraj org jedinice po lokaciji",
                "Odjeli grupirani po veličini"
            ],
            "_ProjectTo": [
                "Daj mi org jedinice s kolonama: naziv, šifra",
                "Lista odjela samo s osnovnim podacima"
            ],
            "_metadata": [
                "Metapodaci za org jedinicu",
                "Dodatne informacije o odjelu"
            ],
            "_DeleteByCriteria": [
                "Obriši neaktivne org jedinice",
                "Masovno brisanje odjela"
            ],
            "_multipatch": [
                "Masovno ažuriraj org jedinice",
                "Ažuriraj više odjela odjednom"
            ]
        }
    },
    "Teams": {
        "hr_name": "timovi",
        "hr_singular": "tim",
        "hr_genitive": "timova",
        "synonyms": ["tim", "grupa", "ekipa", "team"],
        "keywords": ["naziv", "voditelj", "članovi", "projekt"],
        "examples": {
            "_Agg": [
                "Statistika timova",
                "Prosječna veličina tima",
                "Agregirani podaci timova"
            ],
            "_GroupBy": [
                "Grupiraj timove po projektu",
                "Timovi grupirani po veličini"
            ],
            "_ProjectTo": [
                "Daj mi timove s kolonama: naziv, voditelj",
                "Lista timova samo s osnovnim podacima"
            ],
            "_metadata": [
                "Metapodaci za tim",
                "Dodatne informacije o timu"
            ],
            "_DeleteByCriteria": [
                "Obriši neaktivne timove",
                "Masovno brisanje timova"
            ],
            "_multipatch": [
                "Masovno ažuriraj timove",
                "Ažuriraj više timova odjednom"
            ]
        }
    },
    "VehicleCalendar": {
        "hr_name": "kalendar vozila",
        "hr_singular": "kalendar vozila",
        "hr_genitive": "kalendara vozila",
        "synonyms": ["kalendar vozila", "raspored vozila", "rezervacije vozila", "booking"],
        "keywords": ["datum", "vrijeme", "vozilo", "rezervacija", "dostupnost"],
        "examples": {
            "_Agg": [
                "Statistika rezervacija vozila",
                "Prosječno trajanje rezervacije",
                "Iskorištenost flote"
            ],
            "_GroupBy": [
                "Grupiraj rezervacije po vozilu",
                "Bookings grupirani po mjesecu"
            ],
            "_ProjectTo": [
                "Daj mi kalendar vozila s kolonama: datum, vozilo, status",
                "Lista rezervacija samo s osnovnim podacima"
            ],
            "_metadata": [
                "Metapodaci za rezervaciju vozila",
                "Dodatne informacije o bookingu"
            ],
            "_DeleteByCriteria": [
                "Obriši stare rezervacije",
                "Masovno brisanje bookinga po kriteriju"
            ],
            "_multipatch": [
                "Masovno ažuriraj rezervacije",
                "Ažuriraj više bookinga odjednom"
            ]
        }
    },
    "MileageReports": {
        "hr_name": "izvještaji o kilometraži",
        "hr_singular": "izvještaj o kilometraži",
        "hr_genitive": "izvještaja o kilometraži",
        "synonyms": ["kilometraža", "mileage", "prijeđeni km", "km izvještaj"],
        "keywords": ["kilometri", "datum", "vozilo", "vozač"],
        "examples": {
            "_Agg": [
                "Ukupna kilometraža flote",
                "Prosječna dnevna kilometraža",
                "Statistika prijeđenih kilometara"
            ],
            "_GroupBy": [
                "Grupiraj kilometražu po vozilu",
                "Mileage grupiran po mjesecu"
            ],
            "_ProjectTo": [
                "Daj mi kilometražu s kolonama: datum, km, vozilo",
                "Lista km izvještaja samo s osnovnim podacima"
            ],
            "_metadata": [
                "Metapodaci za km izvještaj",
                "Dodatne informacije o kilometraži"
            ],
            "_DeleteByCriteria": [
                "Obriši stare km izvještaje",
                "Masovno brisanje mileage podataka"
            ],
            "_multipatch": [
                "Masovno ažuriraj kilometražu",
                "Ažuriraj više km izvještaja odjednom"
            ]
        }
    },
    "DocumentTypes": {
        "hr_name": "tipovi dokumenata",
        "hr_singular": "tip dokumenta",
        "hr_genitive": "tipova dokumenata",
        "synonyms": ["tip dokumenta", "vrste dokumenata", "kategorija dokumenta", "document type"],
        "keywords": ["naziv", "opis", "kategorija", "tip"],
        "examples": {
            "_Agg": [
                "Statistika tipova dokumenata",
                "Koliko ima tipova dokumenata",
                "Agregirani podaci o vrstama dokumenata"
            ],
            "_GroupBy": [
                "Grupiraj tipove dokumenata po kategoriji",
                "Vrste dokumenata grupirane po statusu"
            ],
            "_ProjectTo": [
                "Daj mi tipove dokumenata s kolonama: naziv, opis",
                "Lista vrsta dokumenata samo s osnovnim podacima"
            ],
            "_metadata": [
                "Metapodaci za tip dokumenta",
                "Dodatne informacije o vrsti dokumenta"
            ],
            "_DeleteByCriteria": [
                "Obriši neaktivne tipove dokumenata",
                "Masovno brisanje vrsta dokumenata"
            ],
            "_multipatch": [
                "Masovno ažuriraj tipove dokumenata",
                "Ažuriraj više vrsta dokumenata odjednom"
            ]
        }
    },
    "Metadata": {
        "hr_name": "metapodaci",
        "hr_singular": "metapodatak",
        "hr_genitive": "metapodataka",
        "synonyms": ["metapodaci", "meta informacije", "dodatni podaci", "prošireni podaci"],
        "keywords": ["ključ", "vrijednost", "atribut", "svojstvo"],
        "examples": {
            "_Agg": [
                "Statistika metapodataka",
                "Agregirani metapodaci",
                "Broj metapodataka po tipu"
            ],
            "_GroupBy": [
                "Grupiraj metapodatke po ključu",
                "Metapodaci grupirani po kategoriji"
            ],
            "_ProjectTo": [
                "Daj mi metapodatke s kolonama: ključ, vrijednost",
                "Lista metapodataka samo s osnovnim podacima"
            ],
            "_metadata": [
                "Dodatni metapodaci za metapodatke",
                "Proširene informacije o metapodacima"
            ],
            "_DeleteByCriteria": [
                "Obriši stare metapodatke",
                "Masovno brisanje metapodataka po kriteriju"
            ],
            "_multipatch": [
                "Masovno ažuriraj metapodatke",
                "Ažuriraj više metapodataka odjednom"
            ]
        }
    },
    "LatestPersonPeriodicActivities": {
        "hr_name": "najnovije periodične aktivnosti osobe",
        "hr_singular": "najnovija periodična aktivnost osobe",
        "hr_genitive": "najnovijih periodičnih aktivnosti osobe",
        "synonyms": ["najnovija aktivnost", "zadnja periodična aktivnost", "latest activity", "osobne aktivnosti"],
        "keywords": ["osoba", "aktivnost", "periodična", "najnovija", "datum"],
        "examples": {
            "_Agg": [
                "Statistika najnovijih aktivnosti osoba",
                "Prosječan broj najnovijih periodičnih aktivnosti",
                "Agregirani podaci o osobnim aktivnostima"
            ],
            "_GroupBy": [
                "Grupiraj najnovije aktivnosti po osobi",
                "Periodične aktivnosti grupirane po tipu"
            ],
            "_ProjectTo": [
                "Daj mi najnovije aktivnosti s kolonama: osoba, tip, datum",
                "Lista periodičnih aktivnosti osoba"
            ],
            "_metadata": [
                "Metapodaci za najnoviju aktivnost osobe",
                "Dodatne informacije o periodičnoj aktivnosti"
            ],
            "_DeleteByCriteria": [
                "Obriši stare najnovije aktivnosti",
                "Masovno brisanje periodičnih aktivnosti osoba"
            ],
            "_multipatch": [
                "Masovno ažuriraj najnovije aktivnosti",
                "Ažuriraj više periodičnih aktivnosti odjednom"
            ]
        }
    },
    "PersonPeriodicActivities": {
        "hr_name": "periodične aktivnosti osobe",
        "hr_singular": "periodična aktivnost osobe",
        "hr_genitive": "periodičnih aktivnosti osobe",
        "synonyms": ["periodična aktivnost", "ponavljajuća aktivnost", "osobna aktivnost", "periodic activity"],
        "keywords": ["osoba", "aktivnost", "periodična", "ponavljanje", "datum"],
        "examples": {
            "_Agg": [
                "Statistika periodičnih aktivnosti osoba",
                "Prosječan broj aktivnosti po osobi",
                "Agregirani podaci o ponavljajućim aktivnostima"
            ],
            "_GroupBy": [
                "Grupiraj periodične aktivnosti po osobi",
                "Aktivnosti osoba grupirane po tipu"
            ],
            "_ProjectTo": [
                "Daj mi periodične aktivnosti s kolonama: osoba, tip, datum",
                "Lista ponavljajućih aktivnosti osoba"
            ],
            "_metadata": [
                "Metapodaci za periodičnu aktivnost osobe",
                "Dodatne informacije o aktivnosti"
            ],
            "_DeleteByCriteria": [
                "Obriši stare periodične aktivnosti",
                "Masovno brisanje aktivnosti osoba po kriteriju"
            ],
            "_multipatch": [
                "Masovno ažuriraj periodične aktivnosti",
                "Ažuriraj više aktivnosti osoba odjednom"
            ]
        }
    },
    "PeriodicActivityTypes": {
        "hr_name": "tipovi periodičnih aktivnosti",
        "hr_singular": "tip periodične aktivnosti",
        "hr_genitive": "tipova periodičnih aktivnosti",
        "synonyms": ["tip aktivnosti", "vrsta aktivnosti", "kategorija aktivnosti"],
        "keywords": ["naziv", "opis", "tip", "kategorija"],
        "examples": {
            "_Agg": [
                "Statistika tipova aktivnosti",
                "Broj tipova periodičnih aktivnosti"
            ],
            "_GroupBy": [
                "Grupiraj tipove aktivnosti po kategoriji"
            ],
            "_ProjectTo": [
                "Daj mi tipove aktivnosti s kolonama: naziv, opis"
            ],
            "_metadata": [
                "Metapodaci za tip aktivnosti"
            ],
            "_DeleteByCriteria": [
                "Obriši neaktivne tipove aktivnosti"
            ],
            "_multipatch": [
                "Masovno ažuriraj tipove aktivnosti"
            ]
        }
    },
    "Master": {
        "hr_name": "master podaci",
        "hr_singular": "master podatak",
        "hr_genitive": "master podataka",
        "synonyms": ["master", "osnovni podaci", "glavni podaci", "master data"],
        "keywords": ["vozilo", "osoba", "osnovni", "master"],
        "examples": {
            "_Agg": [
                "Statistika master podataka",
                "Agregirani master podaci",
                "Prosjeci iz master tablice"
            ],
            "_GroupBy": [
                "Grupiraj master podatke po tipu",
                "Master podaci grupirani po kategoriji"
            ],
            "_ProjectTo": [
                "Daj mi master podatke s određenim kolonama",
                "Lista master podataka samo s osnovnim podacima"
            ],
            "_metadata": [
                "Metapodaci za master podatke",
                "Dodatne informacije o master tablici"
            ],
            "_DeleteByCriteria": [
                "Obriši stare master podatke",
                "Masovno brisanje master podataka"
            ],
            "_multipatch": [
                "Masovno ažuriraj master podatke",
                "Ažuriraj više master zapisa odjednom"
            ]
        }
    },
    "Pools": {
        "hr_name": "poolovi",
        "hr_singular": "pool",
        "hr_genitive": "poolova",
        "synonyms": ["pool", "bazen vozila", "grupa vozila", "vehicle pool"],
        "keywords": ["naziv", "vozila", "kapacitet", "lokacija"],
        "examples": {
            "_Agg": [
                "Statistika poolova vozila",
                "Prosječan kapacitet poola"
            ],
            "_GroupBy": [
                "Grupiraj poolove po lokaciji",
                "Poolovi grupirani po veličini"
            ],
            "_ProjectTo": [
                "Daj mi poolove s kolonama: naziv, lokacija",
                "Lista poolova vozila"
            ],
            "_metadata": [
                "Metapodaci za pool",
                "Dodatne informacije o poolu vozila"
            ],
            "_DeleteByCriteria": [
                "Obriši neaktivne poolove",
                "Masovno brisanje poolova"
            ],
            "_multipatch": [
                "Masovno ažuriraj poolove",
                "Ažuriraj više poolova odjednom"
            ]
        }
    },
    "Tags": {
        "hr_name": "tagovi",
        "hr_singular": "tag",
        "hr_genitive": "tagova",
        "synonyms": ["tag", "oznaka", "label", "kategorija"],
        "keywords": ["naziv", "boja", "kategorija"],
        "examples": {
            "_Agg": [
                "Statistika tagova",
                "Broj tagova po kategoriji"
            ],
            "_GroupBy": [
                "Grupiraj tagove po boji",
                "Oznake grupirane po kategoriji"
            ],
            "_ProjectTo": [
                "Daj mi tagove s kolonama: naziv, boja",
                "Lista oznaka"
            ],
            "_metadata": [
                "Metapodaci za tag",
                "Dodatne informacije o oznaci"
            ],
            "_DeleteByCriteria": [
                "Obriši nekorištene tagove",
                "Masovno brisanje oznaka"
            ],
            "_multipatch": [
                "Masovno ažuriraj tagove",
                "Ažuriraj više oznaka odjednom"
            ]
        }
    },
    "Documents": {
        "hr_name": "dokumenti",
        "hr_singular": "dokument",
        "hr_genitive": "dokumenata",
        "synonyms": ["dokument", "datoteka", "privitak", "file", "attachment"],
        "keywords": ["naziv", "tip", "veličina", "datum"],
        "examples": {
            "_Agg": [
                "Statistika dokumenata",
                "Prosječna veličina dokumenata",
                "Ukupan broj datoteka"
            ],
            "_GroupBy": [
                "Grupiraj dokumente po tipu",
                "Datoteke grupirane po veličini"
            ],
            "_ProjectTo": [
                "Daj mi dokumente s kolonama: naziv, tip, veličina",
                "Lista datoteka samo s osnovnim podacima"
            ],
            "_metadata": [
                "Metapodaci za dokument",
                "Dodatne informacije o datoteci"
            ],
            "_DeleteByCriteria": [
                "Obriši stare dokumente",
                "Masovno brisanje datoteka po kriteriju"
            ],
            "_multipatch": [
                "Masovno ažuriraj dokumente",
                "Ažuriraj više datoteka odjednom"
            ]
        }
    }
}

# Generic suffix patterns
SUFFIX_PATTERNS = ["_Agg", "_GroupBy", "_ProjectTo", "_metadata", "_DeleteByCriteria", "_multipatch"]

# Generic purpose templates per suffix
PURPOSE_TEMPLATES = {
    "_Agg": "Ovaj endpoint omogućuje dohvaćanje AGREGIRANIH vrijednosti za {entity_hr} (prosjek, suma, min, max).",
    "_GroupBy": "Ovaj endpoint omogućuje GRUPIRANJE {entity_genitive} po određenom kriteriju.",
    "_ProjectTo": "Ovaj endpoint omogućuje dohvaćanje {entity_genitive} s ODABRANIM kolonama (projekcija).",
    "_metadata": "Ovaj endpoint omogućuje dohvaćanje METAPODATAKA za {entity_singular_hr}.",
    "_DeleteByCriteria": "Ovaj endpoint omogućuje MASOVNO BRISANJE {entity_genitive} po zadanom kriteriju.",
    "_multipatch": "Ovaj endpoint omogućuje MASOVNO AŽURIRANJE više {entity_genitive} odjednom."
}

# When to use templates
WHEN_TO_USE_TEMPLATES = {
    "_Agg": [
        "Kada želite izračunati statistiku za {entity_hr}.",
        "Kada trebate prosječne, ukupne ili ekstremne vrijednosti."
    ],
    "_GroupBy": [
        "Kada želite grupirati {entity_hr} po određenom polju.",
        "Kada trebate agregaciju s grupiranjem."
    ],
    "_ProjectTo": [
        "Kada želite dohvatiti samo određene kolone {entity_genitive}.",
        "Kada ne trebate sve podatke, samo projekciju."
    ],
    "_metadata": [
        "Kada želite vidjeti dodatne informacije o {entity_singular_hr}.",
        "Kada trebate proširene metapodatke."
    ],
    "_DeleteByCriteria": [
        "Kada želite obrisati više {entity_genitive} odjednom.",
        "Kada trebate bulk delete po kriteriju."
    ],
    "_multipatch": [
        "Kada želite ažurirati više {entity_genitive} odjednom.",
        "Kada trebate bulk update."
    ]
}


def extract_entity_from_tool_name(tool_name: str) -> str:
    """Extract entity name from tool name like 'get_Vehicles_Agg' -> 'Vehicles'"""
    parts = tool_name.split("_")
    if len(parts) >= 2:
        # Skip HTTP method prefix (get, post, put, patch, delete)
        if parts[0] in ["get", "post", "put", "patch", "delete"]:
            return parts[1]
    return ""


def get_entity_mapping(entity: str) -> Dict:
    """Get entity mapping, with fallback to generic if not found."""
    # Direct match
    if entity in ENTITY_MAPPINGS:
        return ENTITY_MAPPINGS[entity]

    # Try case-insensitive match
    for key, value in ENTITY_MAPPINGS.items():
        if key.lower() == entity.lower():
            return value

    # Fallback - create generic mapping
    return {
        "hr_name": entity.lower(),
        "hr_singular": entity.lower(),
        "hr_genitive": entity.lower(),
        "synonyms": [entity.lower()],
        "keywords": [],
        "examples": {}
    }


def is_generic_query(query: str, entity_keywords: List[str]) -> bool:
    """Check if a query is generic (doesn't mention specific entity)."""
    query_lower = query.lower()

    # Generic patterns that indicate no specific entity
    generic_patterns = [
        "polje x", "polje y", "polje z",
        "po polju", "za polje",
        "s id-om", "id-em",
        "po određenom", "za određeno",
        "za određen", "određeni",
        "s navedenim", "navedeni",
        "po zadanom", "zadani",
    ]

    for pattern in generic_patterns:
        if pattern in query_lower:
            return True

    # If no entity keywords are mentioned, it's generic
    has_entity_mention = False
    for keyword in entity_keywords:
        if keyword.lower() in query_lower:
            has_entity_mention = True
            break

    return not has_entity_mention


def update_tool_documentation(tool_data: Dict, suffix: str, entity: str, entity_mapping: Dict) -> Dict:
    """Update a single tool's documentation with entity-specific content."""

    updated = tool_data.copy()

    # Get examples for this suffix
    examples = entity_mapping.get("examples", {}).get(suffix, [])

    # Get entity keywords for filtering
    entity_keywords = entity_mapping.get("synonyms", []) + [entity.lower()]

    # Update purpose with entity-specific text
    if suffix in PURPOSE_TEMPLATES:
        updated["purpose"] = PURPOSE_TEMPLATES[suffix].format(
            entity_hr=entity_mapping.get("hr_name", entity.lower()),
            entity_genitive=entity_mapping.get("hr_genitive", entity.lower()),
            entity_singular_hr=entity_mapping.get("hr_singular", entity.lower())
        )

    # Update when_to_use with entity-specific text
    if suffix in WHEN_TO_USE_TEMPLATES:
        updated["when_to_use"] = [
            template.format(
                entity_hr=entity_mapping.get("hr_name", entity.lower()),
                entity_genitive=entity_mapping.get("hr_genitive", entity.lower()),
                entity_singular_hr=entity_mapping.get("hr_singular", entity.lower())
            )
            for template in WHEN_TO_USE_TEMPLATES[suffix]
        ]

    # REPLACE example_queries_hr - remove generic, add entity-specific
    if examples:
        existing_examples = updated.get("example_queries_hr", [])

        # Filter out generic examples from existing
        filtered_existing = [
            e for e in existing_examples
            if not is_generic_query(e, entity_keywords)
        ]

        # New examples first, then filtered existing
        all_examples = examples + [e for e in filtered_existing if e not in examples]
        updated["example_queries_hr"] = all_examples[:6]  # Max 6 examples

    # Add/update synonyms_hr with entity-specific synonyms
    entity_synonyms = entity_mapping.get("synonyms", [])
    suffix_synonyms = {
        "_Agg": ["agregacija", "statistika", "prosjek", "suma", "avg", "max", "min"],
        "_GroupBy": ["grupiraj", "grupiranje", "group by", "po grupama"],
        "_ProjectTo": ["projekcija", "kolone", "polja", "atributi", "filtar kolona"],
        "_metadata": ["metapodaci", "meta", "dodatne info", "prošireni podaci"],
        "_DeleteByCriteria": ["masovno brisanje", "bulk delete", "obriši više", "brisanje po kriteriju"],
        "_multipatch": ["masovno ažuriranje", "bulk update", "ažuriraj više", "batch update"]
    }

    combined_synonyms = []
    # Entity + suffix synonyms
    for entity_syn in entity_synonyms[:3]:  # Top 3 entity synonyms
        for suffix_syn in suffix_synonyms.get(suffix, [])[:2]:  # Top 2 suffix synonyms
            combined_synonyms.append(f"{suffix_syn} {entity_syn}")

    # Add pure entity synonyms
    combined_synonyms.extend(entity_synonyms)
    # Add pure suffix synonyms
    combined_synonyms.extend(suffix_synonyms.get(suffix, []))

    existing_synonyms = updated.get("synonyms_hr", [])
    all_synonyms = combined_synonyms + [s for s in existing_synonyms if s not in combined_synonyms]
    updated["synonyms_hr"] = all_synonyms[:12]  # Max 12 synonyms

    # Add disambiguation keywords
    updated["disambiguation_keywords"] = entity_synonyms + entity_mapping.get("keywords", [])

    # Mark as updated
    updated["_updated_at"] = datetime.now().isoformat()
    updated["_documentation_version"] = "2.0"

    return updated


def main():
    """Main function to update tool documentation."""
    print("=" * 70)
    print("FIX GENERIC TOOL DOCUMENTATION")
    print("=" * 70)

    # Load existing documentation
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
    doc_path = os.path.join(config_dir, "tool_documentation.json")

    print(f"\nLoading: {doc_path}")

    with open(doc_path, 'r', encoding='utf-8') as f:
        documentation = json.load(f)

    print(f"Loaded {len(documentation)} tools")

    # Track updates
    updates_by_suffix = {suffix: 0 for suffix in SUFFIX_PATTERNS}
    updated_tools = []

    # Process each tool
    for tool_name, tool_data in documentation.items():
        # Check if this is a generic tool (has one of the suffixes)
        for suffix in SUFFIX_PATTERNS:
            if suffix in tool_name:
                entity = extract_entity_from_tool_name(tool_name)
                if entity:
                    entity_mapping = get_entity_mapping(entity)

                    # Update the documentation
                    documentation[tool_name] = update_tool_documentation(
                        tool_data, suffix, entity, entity_mapping
                    )

                    updates_by_suffix[suffix] += 1
                    updated_tools.append(tool_name)
                break  # Only match first suffix

    # Save updated documentation
    print(f"\nSaving updated documentation...")
    with open(doc_path, 'w', encoding='utf-8') as f:
        json.dump(documentation, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("UPDATE SUMMARY")
    print("=" * 70)

    total_updated = 0
    for suffix, count in updates_by_suffix.items():
        print(f"  {suffix}: {count} tools updated")
        total_updated += count

    print(f"\n  TOTAL: {total_updated} tools updated")
    print("=" * 70)

    # Show sample of updated tools
    print("\nSample updated tools:")
    for tool in updated_tools[:10]:
        print(f"  - {tool}")
    if len(updated_tools) > 10:
        print(f"  ... and {len(updated_tools) - 10} more")

    print("\n[SUCCESS] Documentation updated!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
