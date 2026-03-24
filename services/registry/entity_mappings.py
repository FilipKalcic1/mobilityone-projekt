"""
Entity mappings for Croatian language support in embedding generation.

Contains dictionaries mapping English API terms to Croatian translations,
synonyms, and segments to skip during path parsing.

These constants are used by EmbeddingEngine to generate Croatian-language
embeddings from English API definitions.
"""

# Comprehensive entity mappings for path/operationId extraction
PATH_ENTITY_MAP = {
    # Vehicles & Fleet
    "vehicle": ("vozilo", "vozila"),
    "vehicles": ("vozilo", "vozila"),
    "car": ("automobil", "automobila"),
    "fleet": ("flota", "flote"),
    "fleets": ("flota", "flote"),
    "asset": ("imovina", "imovine"),
    "assets": ("imovina", "imovine"),
    # People
    "person": ("osoba", "osobe"),
    "persons": ("osoba", "osobe"),
    "people": ("osoba", "osobe"),
    "driver": ("vozač", "vozača"),
    "drivers": ("vozač", "vozača"),
    "user": ("korisnik", "korisnika"),
    "users": ("korisnik", "korisnika"),
    "employee": ("zaposlenik", "zaposlenika"),
    "employees": ("zaposlenik", "zaposlenika"),
    "customer": ("kupac", "kupca"),
    "customers": ("kupac", "kupaca"),
    "contact": ("kontakt", "kontakta"),
    "contacts": ("kontakt", "kontakata"),
    # Bookings & Reservations
    "booking": ("rezervacija", "rezervacije"),
    "bookings": ("rezervacija", "rezervacije"),
    "reservation": ("rezervacija", "rezervacije"),
    "reservations": ("rezervacija", "rezervacija"),
    "calendar": ("kalendar", "kalendara"),
    "appointment": ("termin", "termina"),
    # Locations
    "location": ("lokacija", "lokacije"),
    "locations": ("lokacija", "lokacija"),
    "address": ("adresa", "adrese"),
    "zone": ("zona", "zone"),
    "zones": ("zona", "zona"),
    "region": ("regija", "regije"),
    "branch": ("poslovnica", "poslovnice"),
    "branches": ("poslovnica", "poslovnica"),
    "station": ("stanica", "stanice"),
    "depot": ("depo", "depoa"),
    # Documents & Records
    "document": ("dokument", "dokumenta"),
    "documents": ("dokument", "dokumenata"),
    "invoice": ("račun", "računa"),
    "invoices": ("račun", "računa"),
    "contract": ("ugovor", "ugovora"),
    "contracts": ("ugovor", "ugovora"),
    "report": ("izvještaj", "izvještaja"),
    "reports": ("izvještaj", "izvještaja"),
    "log": ("zapis", "zapisa"),
    "logs": ("zapis", "zapisa"),
    "history": ("povijest", "povijesti"),
    "record": ("zapis", "zapisa"),
    # Maintenance & Service
    "maintenance": ("održavanje", "održavanja"),
    "service": ("servis", "servisa"),
    "repair": ("popravak", "popravka"),
    "inspection": ("inspekcija", "inspekcije"),
    "damage": ("šteta", "štete"),
    "damages": ("šteta", "šteta"),
    "accident": ("nesreća", "nesreće"),
    "insurance": ("osiguranje", "osiguranja"),
    # Financial
    "payment": ("plaćanje", "plaćanja"),
    "payments": ("plaćanje", "plaćanja"),
    "cost": ("trošak", "troška"),
    "costs": ("trošak", "troškova"),
    "expense": ("trošak", "troška"),
    "expenses": ("trošak", "troškova"),
    "fee": ("naknada", "naknade"),
    "fees": ("naknada", "naknada"),
    "price": ("cijena", "cijene"),
    "pricing": ("cjenik", "cjenika"),
    "tariff": ("tarifa", "tarife"),
    "billing": ("naplata", "naplate"),
    "transaction": ("transakcija", "transakcije"),
    # Status & Metrics
    "status": ("status", "statusa"),
    "state": ("stanje", "stanja"),
    "mileage": ("kilometraža", "kilometraže"),
    "odometer": ("kilometraža", "kilometraže"),
    "fuel": ("gorivo", "goriva"),
    "battery": ("baterija", "baterije"),
    "tire": ("guma", "gume"),
    "tires": ("guma", "guma"),
    "oil": ("ulje", "ulja"),
    # Access & Permissions
    "permission": ("dozvola", "dozvole"),
    "permissions": ("dozvola", "dozvola"),
    "role": ("uloga", "uloge"),
    "roles": ("uloga", "uloga"),
    "access": ("pristup", "pristupa"),
    "group": ("grupa", "grupe"),
    "groups": ("grupa", "grupa"),
    "team": ("tim", "tima"),
    "department": ("odjel", "odjela"),
    # Equipment & Accessories
    "equipment": ("oprema", "opreme"),
    "accessory": ("dodatak", "dodatka"),
    "accessories": ("dodatak", "dodataka"),
    "device": ("uređaj", "uređaja"),
    "devices": ("uređaj", "uređaja"),
    "tracker": ("tracker", "trackera"),
    "gps": ("GPS", "GPS-a"),
    "telematics": ("telematika", "telematike"),
    # Categories & Types
    "category": ("kategorija", "kategorije"),
    "categories": ("kategorija", "kategorija"),
    "type": ("tip", "tipa"),
    "types": ("tip", "tipova"),
    "class": ("klasa", "klase"),
    "brand": ("marka", "marke"),
    "model": ("model", "modela"),
    # Time-related
    "period": ("period", "perioda"),
    "schedule": ("raspored", "rasporeda"),
    "shift": ("smjena", "smjene"),
    "availability": ("dostupnost", "dostupnosti"),
    "slot": ("termin", "termina"),
    # Misc
    "notification": ("obavijest", "obavijesti"),
    "notifications": ("obavijest", "obavijesti"),
    "alert": ("upozorenje", "upozorenja"),
    "alerts": ("upozorenje", "upozorenja"),
    "message": ("poruka", "poruke"),
    "note": ("bilješka", "bilješke"),
    "notes": ("bilješka", "bilješki"),
    "comment": ("komentar", "komentara"),
    "tag": ("oznaka", "oznake"),
    "tags": ("oznaka", "oznaka"),
    "image": ("slika", "slike"),
    "images": ("slika", "slika"),
    "photo": ("fotografija", "fotografije"),
    "file": ("datoteka", "datoteke"),
    "attachment": ("privitak", "privitka"),
    "tenant": ("najmodavac", "najmodavca"),
    "organization": ("organizacija", "organizacije"),
    "company": ("tvrtka", "tvrtke"),
    "license": ("licenca", "licence"),
    "registration": ("registracija", "registracije"),
    "certificate": ("certifikat", "certifikata"),
    "policy": ("polica", "police"),
    "claim": ("zahtjev", "zahtjeva"),
    "request": ("zahtjev", "zahtjeva"),
    "order": ("narudžba", "narudžbe"),
    "pool": ("bazen", "bazena"),
    "trip": ("putovanje", "putovanja"),
    "route": ("ruta", "rute"),
    "journey": ("vožnja", "vožnje"),
    "ride": ("vožnja", "vožnje"),
    "transfer": ("transfer", "transfera"),
    "pickup": ("preuzimanje", "preuzimanja"),
    "dropoff": ("vraćanje", "vraćanja"),
    "checkin": ("prijava", "prijave"),
    "checkout": ("odjava", "odjave"),
    "handover": ("primopredaja", "primopredaje"),
    "key": ("ključ", "ključa"),
    "keys": ("ključ", "ključeva"),
    "card": ("kartica", "kartice"),
    "fuelcard": ("kartica za gorivo", "kartice za gorivo"),
    "tollcard": ("ENC kartica", "ENC kartice"),
    "violation": ("prekršaj", "prekršaja"),
    "fine": ("kazna", "kazne"),
    "penalty": ("kazna", "kazne"),
    # Additional common API terms (v3.2 expansion)
    "summary": ("sažetak", "sažetka"),
    "detail": ("detalj", "detalja"),
    "details": ("detalji", "detalja"),
    "info": ("informacija", "informacije"),
    "information": ("informacija", "informacije"),
    "data": ("podaci", "podataka"),
    "list": ("lista", "liste"),
    "item": ("stavka", "stavke"),
    "items": ("stavke", "stavki"),
    "entry": ("unos", "unosa"),
    "entries": ("unosi", "unosa"),
    "event": ("događaj", "događaja"),
    "events": ("događaji", "događaja"),
    "activity": ("aktivnost", "aktivnosti"),
    "activities": ("aktivnosti", "aktivnosti"),
    "action": ("akcija", "akcije"),
    "task": ("zadatak", "zadatka"),
    "tasks": ("zadaci", "zadataka"),
    "job": ("posao", "posla"),
    "workflow": ("tijek rada", "tijeka rada"),
    "process": ("proces", "procesa"),
    "settings": ("postavke", "postavki"),
    "configuration": ("konfiguracija", "konfiguracije"),
    "config": ("konfiguracija", "konfiguracije"),
    "preference": ("postavka", "postavke"),
    "option": ("opcija", "opcije"),
    "options": ("opcije", "opcija"),
    "filter": ("filter", "filtera"),
    "search": ("pretraga", "pretrage"),
    "query": ("upit", "upita"),
    "result": ("rezultat", "rezultata"),
    "results": ("rezultati", "rezultata"),
    "response": ("odgovor", "odgovora"),
    "error": ("greška", "greške"),
    "warning": ("upozorenje", "upozorenja"),
    "audit": ("revizija", "revizije"),
    "export": ("izvoz", "izvoza"),
    "import": ("uvoz", "uvoza"),
    "download": ("preuzimanje", "preuzimanja"),
    "upload": ("učitavanje", "učitavanja"),
    "sync": ("sinkronizacija", "sinkronizacije"),
    "backup": ("sigurnosna kopija", "sigurnosne kopije"),
    "archive": ("arhiva", "arhive"),
    "version": ("verzija", "verzije"),
    "revision": ("revizija", "revizije"),
    "change": ("promjena", "promjene"),
    "changes": ("promjene", "promjena"),
    "update": ("ažuriranje", "ažuriranja"),
    "statistics": ("statistika", "statistike"),
    "analytics": ("analitika", "analitike"),
    "metric": ("metrika", "metrike"),
    "metrics": ("metrike", "metrika"),
    "dashboard": ("nadzorna ploča", "nadzorne ploče"),
    "overview": ("pregled", "pregleda"),
    "chart": ("grafikon", "grafikona"),
    "graph": ("graf", "grafa"),
    "utilization": ("iskorištenost", "iskorištenosti"),
    "usage": ("korištenje", "korištenja"),
    "consumption": ("potrošnja", "potrošnje"),
    "rate": ("stopa", "stope"),
    "ratio": ("omjer", "omjera"),
    "percentage": ("postotak", "postotka"),
    "count": ("broj", "broja"),
    "total": ("ukupno", "ukupnog"),
    "average": ("prosjek", "prosjeka"),
    "minimum": ("minimum", "minimuma"),
    "maximum": ("maksimum", "maksimuma"),
    # Additional API patterns
    "rental": ("najam", "najma"),
    "rentals": ("najam", "najmova"),
    "rent": ("najam", "najma"),
    "rents": ("najam", "najmova"),
    "hire": ("najam", "najma"),
    "lease": ("leasing", "leasinga"),
    "leasing": ("leasing", "leasinga"),
    "credit": ("kredit", "kredita"),
    "debit": ("zaduženje", "zaduženja"),
    "refund": ("povrat", "povrata"),
    "refunds": ("povrat", "povrata"),
    "cancellation": ("otkazivanje", "otkazivanja"),
    "cancel": ("otkazivanje", "otkazivanja"),
    "extend": ("produženje", "produženja"),
    "extension": ("produženje", "produženja"),
    "modify": ("izmjena", "izmjene"),
    "modification": ("izmjena", "izmjene"),
    "quote": ("ponuda", "ponude"),
    "quotes": ("ponuda", "ponuda"),
    "estimate": ("procjena", "procjene"),
    "offer": ("ponuda", "ponude"),
    "offers": ("ponuda", "ponuda"),
    "promo": ("promocija", "promocije"),
    "promotion": ("promocija", "promocije"),
    "promotions": ("promocija", "promocija"),
    "coupon": ("kupon", "kupona"),
    "coupons": ("kupon", "kupona"),
    "voucher": ("vaučer", "vaučera"),
    "vouchers": ("vaučer", "vaučera"),
    "loyalty": ("lojalnost", "lojalnosti"),
    "points": ("bodovi", "bodova"),
    "reward": ("nagrada", "nagrade"),
    "rewards": ("nagrada", "nagrada"),
    "tier": ("razina", "razine"),
    "level": ("razina", "razine"),
    "membership": ("članstvo", "članstva"),
    "member": ("član", "člana"),
    "members": ("član", "članova"),
    "subscription": ("pretplata", "pretplate"),
    "subscriber": ("pretplatnik", "pretplatnika"),
    "plan": ("plan", "plana"),
    "plans": ("plan", "planova"),
    "package": ("paket", "paketa"),
    "packages": ("paket", "paketa"),
    "addon": ("dodatak", "dodatka"),
    "addons": ("dodatak", "dodataka"),
    "extra": ("dodatak", "dodatka"),
    "extras": ("dodatak", "dodataka"),
    "feature": ("značajka", "značajke"),
    "features": ("značajka", "značajki"),
    "spec": ("specifikacija", "specifikacije"),
    "specification": ("specifikacija", "specifikacije"),
    "specifications": ("specifikacija", "specifikacija"),
    "attribute": ("atribut", "atributa"),
    "attributes": ("atribut", "atributa"),
    "property": ("svojstvo", "svojstva"),
    "properties": ("svojstvo", "svojstava"),
    "metadata": ("metapodaci", "metapodataka"),
    "meta": ("metapodaci", "metapodataka"),
    "custom": ("prilagođeno", "prilagođenog"),
    "standard": ("standardno", "standardnog"),
    "default": ("zadano", "zadanog"),
    "basic": ("osnovno", "osnovnog"),
    "premium": ("premium", "premium"),
    "vip": ("VIP", "VIP"),
    "economy": ("ekonomija", "ekonomije"),
    "comfort": ("komfort", "komfora"),
    "luxury": ("luksuz", "luksuza"),
    "compact": ("kompaktno", "kompaktnog"),
    "suv": ("SUV", "SUV-a"),
    "van": ("kombi", "kombija"),
    "minivan": ("minivan", "minivana"),
    "truck": ("kamion", "kamiona"),
    "motorcycle": ("motocikl", "motocikla"),
    "scooter": ("skuter", "skutera"),
    "bike": ("bicikl", "bicikla"),
    "electric": ("električno", "električnog"),
    "hybrid": ("hibrid", "hibrida"),
    "diesel": ("dizel", "dizela"),
    "petrol": ("benzin", "benzina"),
    "gasoline": ("benzin", "benzina"),
    "automatic": ("automatik", "automatika"),
    "manual": ("ručni mjenjač", "ručnog mjenjača"),
    "transmission": ("mjenjač", "mjenjača"),
    "engine": ("motor", "motora"),
    "horsepower": ("konjska snaga", "konjske snage"),
    "power": ("snaga", "snage"),
    "torque": ("okretni moment", "okretnog momenta"),
    "cylinder": ("cilindar", "cilindra"),
    "displacement": ("obujam motora", "obujma motora"),
    "acceleration": ("ubrzanje", "ubrzanja"),
    "topspeed": ("maksimalna brzina", "maksimalne brzine"),
    "seat": ("sjedalo", "sjedala"),
    "seats": ("sjedala", "sjedala"),
    "door": ("vrata", "vrata"),
    "doors": ("vrata", "vrata"),
    "luggage": ("prtljaga", "prtljage"),
    "trunk": ("prtljažnik", "prtljažnika"),
    "cargo": ("teret", "tereta"),
    "capacity": ("kapacitet", "kapaciteta"),
    "weight": ("težina", "težine"),
    "dimension": ("dimenzija", "dimenzije"),
    "dimensions": ("dimenzije", "dimenzija"),
    "length": ("dužina", "dužine"),
    "width": ("širina", "širine"),
    "height": ("visina", "visine"),
    "color": ("boja", "boje"),
    "colour": ("boja", "boje"),
    "interior": ("interijer", "interijera"),
    "exterior": ("eksterijer", "eksterijera"),
    "warranty": ("garancija", "garancije"),
    "guarantee": ("garancija", "garancije"),
    "recall": ("opoziv", "opoziva"),
    "recalls": ("opoziv", "opoziva"),
    "compliance": ("usklađenost", "usklađenosti"),
    "emission": ("emisija", "emisije"),
    "emissions": ("emisija", "emisija"),
    "eco": ("eko", "eko"),
    "green": ("zeleno", "zelenog"),
    "sustainable": ("održivo", "održivog"),
    "carbon": ("ugljik", "ugljika"),
    "footprint": ("otisak", "otiska"),
}

# Output key mappings for result description (v3.2 expanded - 200+ entries)
# Maps API output field names to Croatian descriptions
#
# COVERAGE CATEGORIES:
# - Vehicle & Fleet (35 entries)
# - Status & State (20 entries)
# - Documents & Registration (25 entries)
# - Time & Duration (30 entries)
# - Financial (30 entries)
# - Identity & Contact (25 entries)
# - Person Data (20 entries)
# - Booking & Reservation (20 entries)
# - Lists & Counts (15 entries)
# - Technical & System (20 entries)
OUTPUT_KEY_MAP = {
    # ---
    # VEHICLE & FLEET (35 entries)
    # ---
    "mileage": "kilometražu",
    "km": "kilometre",
    "odometer": "stanje kilometara",
    "odometervalue": "vrijednost kilometraže",
    "totalmileage": "ukupnu kilometražu",
    "fuel": "razinu goriva",
    "fuellevel": "razinu goriva",
    "fuelconsumption": "potrošnju goriva",
    "fueltype": "vrstu goriva",
    "fuelcapacity": "kapacitet spremnika",
    "averageconsumption": "prosječnu potrošnju",
    "battery": "stanje baterije",
    "batterylevel": "razinu baterije",
    "batteryvoltage": "napon baterije",
    "chargelevel": "razinu punjenja",
    "speed": "brzinu",
    "averagespeed": "prosječnu brzinu",
    "maxspeed": "maksimalnu brzinu",
    "enginehours": "radne sate motora",
    "enginestatus": "status motora",
    "ignition": "paljenje",
    "doors": "stanje vrata",
    "doorstatus": "status vrata",
    "trunk": "stanje prtljažnika",
    "windows": "stanje prozora",
    "lights": "stanje svjetala",
    "oilpressure": "tlak ulja",
    "oillevel": "razinu ulja",
    "tirePressure": "tlak u gumama",
    "coolanttemperature": "temperaturu rashladne tekućine",
    "vehicletype": "tip vozila",
    "vehicleclass": "klasu vozila",
    "vehiclecategory": "kategoriju vozila",
    "make": "proizvođača",
    "manufacturer": "proizvođača",

    # ---
    # LOCATION & POSITION (15 entries)
    # ---
    "location": "lokaciju",
    "currentlocation": "trenutnu lokaciju",
    "lastlocation": "zadnju lokaciju",
    "position": "poziciju",
    "coordinates": "koordinate",
    "latitude": "geografsku širinu",
    "longitude": "geografsku dužinu",
    "lat": "širinu",
    "lng": "dužinu",
    "altitude": "nadmorsku visinu",
    "heading": "smjer kretanja",
    "direction": "smjer",
    "address": "adresu",
    "fulladdress": "punu adresu",
    "geofence": "geofence zonu",

    # ---
    # STATUS & STATE (20 entries)
    # ---
    "status": "status",
    "state": "stanje",
    "currentstate": "trenutno stanje",
    "available": "dostupnost",
    "availability": "dostupnost",
    "isavailable": "je li dostupno",
    "active": "aktivnost",
    "isactive": "je li aktivno",
    "enabled": "omogućenost",
    "isenabled": "je li omogućeno",
    "locked": "zaključanost",
    "islocked": "je li zaključano",
    "online": "online status",
    "isonline": "je li online",
    "connected": "povezanost",
    "isconnected": "je li povezano",
    "operational": "operativnost",
    "condition": "stanje",
    "health": "zdravlje",
    "healthstatus": "status ispravnosti",

    # ---
    # REGISTRATION & DOCUMENTS (25 entries)
    # ---
    "registration": "registraciju",
    "registrationnumber": "registarsku oznaku",
    "plate": "tablice",
    "licenseplate": "registarske tablice",
    "platenumber": "broj tablica",
    "vin": "broj šasije",
    "chassisnumber": "broj šasije",
    "serialnumber": "serijski broj",
    "expiry": "datum isteka",
    "expirydate": "datum isteka",
    "expiration": "datum isteka",
    "expirationdate": "datum isteka",
    "validuntil": "vrijedi do",
    "validfrom": "vrijedi od",
    "validto": "vrijedi do",
    "issuedate": "datum izdavanja",
    "issuedby": "izdao",
    "issuedto": "izdano za",
    "documentnumber": "broj dokumenta",
    "documenttype": "tip dokumenta",
    "licensenumber": "broj licence",
    "licensetype": "tip licence",
    "licenseclass": "klasa licence",
    "insurancenumber": "broj osiguranja",
    "policynumber": "broj police",

    # ---
    # TIME & DURATION (30 entries)
    # ---
    "date": "datum",
    "time": "vrijeme",
    "datetime": "datum i vrijeme",
    "timestamp": "vremensku oznaku",
    "createdat": "datum kreiranja",
    "createddate": "datum kreiranja",
    "creationdate": "datum nastanka",
    "updatedat": "datum ažuriranja",
    "modifiedat": "datum izmjene",
    "modifieddate": "datum izmjene",
    "lastmodified": "zadnja izmjena",
    "startedat": "vrijeme početka",
    "startdate": "datum početka",
    "starttime": "vrijeme početka",
    "endedat": "vrijeme završetka",
    "enddate": "datum završetka",
    "endtime": "vrijeme završetka",
    "duration": "trajanje",
    "totalduration": "ukupno trajanje",
    "estimatedduration": "procijenjeno trajanje",
    "scheduleddate": "zakazani datum",
    "scheduledtime": "zakazano vrijeme",
    "duedate": "rok",
    "deadline": "krajnji rok",
    "period": "period",
    "timeperiod": "vremenski period",
    "year": "godinu",
    "month": "mjesec",
    "day": "dan",
    "hour": "sat",

    # ---
    # FINANCIAL (30 entries)
    # ---
    "price": "cijenu",
    "unitprice": "jediničnu cijenu",
    "totalprice": "ukupnu cijenu",
    "cost": "trošak",
    "totalcost": "ukupni trošak",
    "amount": "iznos",
    "totalamount": "ukupni iznos",
    "total": "ukupno",
    "subtotal": "međuzbroj",
    "grandtotal": "sveukupno",
    "tax": "porez",
    "taxamount": "iznos poreza",
    "taxrate": "stopu poreza",
    "vat": "PDV",
    "vatamount": "iznos PDV-a",
    "discount": "popust",
    "discountamount": "iznos popusta",
    "discountpercent": "postotak popusta",
    "balance": "stanje računa",
    "deposit": "polog",
    "depositamount": "iznos pologa",
    "refund": "povrat",
    "refundamount": "iznos povrata",
    "payment": "plaćanje",
    "paymentamount": "iznos uplate",
    "paymentstatus": "status plaćanja",
    "paymentmethod": "način plaćanja",
    "currency": "valutu",
    "rate": "stopu",
    "fee": "naknadu",

    # ---
    # IDENTIFICATION (25 entries)
    # ---
    "id": "identifikator",
    "uid": "jedinstveni ID",
    "guid": "globalni ID",
    "externalid": "vanjski ID",
    "internalid": "interni ID",
    "name": "naziv",
    "fullname": "puni naziv",
    "displayname": "prikazni naziv",
    "shortname": "kratki naziv",
    "title": "naslov",
    "description": "opis",
    "summary": "sažetak",
    "details": "detalje",
    "code": "šifru",
    "shortcode": "kratku šifru",
    "number": "broj",
    "reference": "referencu",
    "referencenumber": "referentni broj",
    "identifier": "identifikator",
    "key": "ključ",
    "label": "oznaku",
    "tag": "tag",
    "tags": "tagove",
    "category": "kategoriju",
    "type": "tip",

    # ---
    # CONTACT (15 entries)
    # ---
    "email": "e-mail",
    "emailaddress": "e-mail adresu",
    "phone": "telefon",
    "phonenumber": "broj telefona",
    "mobile": "mobitel",
    "mobilephone": "mobilni telefon",
    "fax": "fax",
    "website": "web stranicu",
    "url": "URL",
    "city": "grad",
    "street": "ulicu",
    "zip": "poštanski broj",
    "zipcode": "poštanski broj",
    "postalcode": "poštanski broj",
    "country": "državu",

    # ---
    # PERSON DATA (20 entries)
    # ---
    "firstname": "ime",
    "lastname": "prezime",
    "middlename": "srednje ime",
    "birthdate": "datum rođenja",
    "dateofbirth": "datum rođenja",
    "age": "dob",
    "gender": "spol",
    "sex": "spol",
    "nationality": "nacionalnost",
    "citizenship": "državljanstvo",
    "personalid": "osobni ID",
    "oib": "OIB",
    "ssn": "matični broj",
    "driverlicense": "vozačku dozvolu",
    "driverlicensenumber": "broj vozačke",
    "licenseexpiry": "istek vozačke",
    "passportnumber": "broj putovnice",
    "idcardnumber": "broj osobne",
    "occupation": "zanimanje",
    "employer": "poslodavca",

    # ---
    # BOOKING & RESERVATION (20 entries)
    # ---
    "bookingid": "ID rezervacije",
    "bookingnumber": "broj rezervacije",
    "bookingcode": "šifru rezervacije",
    "bookingstatus": "status rezervacije",
    "reservationid": "ID rezervacije",
    "reservationnumber": "broj rezervacije",
    "pickupdate": "datum preuzimanja",
    "pickuptime": "vrijeme preuzimanja",
    "pickuplocation": "mjesto preuzimanja",
    "returndate": "datum vraćanja",
    "returntime": "vrijeme vraćanja",
    "returnlocation": "mjesto vraćanja",
    "rentalperiod": "period najma",
    "rentaldays": "dane najma",
    "rentalhours": "sate najma",
    "extras": "dodatke",
    "optionalextras": "opcijske dodatke",
    "insurance": "osiguranje",
    "insurancetype": "tip osiguranja",
    "driver": "vozača",

    # ---
    # LISTS & COUNTS (15 entries)
    # ---
    "count": "broj",
    "totalcount": "ukupan broj",
    "itemcount": "broj stavki",
    "recordcount": "broj zapisa",
    "items": "stavke",
    "list": "popis",
    "results": "rezultate",
    "data": "podatke",
    "records": "zapise",
    "rows": "redove",
    "entries": "unose",
    "page": "stranicu",
    "pagenumber": "broj stranice",
    "pagesize": "veličinu stranice",
    "totalpages": "ukupno stranica",

    # ---
    # TECHNICAL & SYSTEM (20 entries)
    # ---
    "version": "verziju",
    "revision": "reviziju",
    "build": "build",
    "environment": "okruženje",
    "server": "server",
    "client": "klijent",
    "session": "sesiju",
    "token": "token",
    "apikey": "API ključ",
    "secret": "tajni ključ",
    "hash": "hash",
    "checksum": "kontrolnu sumu",
    "signature": "potpis",
    "encoding": "kodiranje",
    "format": "format",
    "mimetype": "MIME tip",
    "contenttype": "tip sadržaja",
    "size": "veličinu",
    "filesize": "veličinu datoteke",
    "length": "dužinu",
}

# Croatian synonyms for common user queries (v3.2 expanded - 60+ groups)
# Uses root forms to match both nominative and genitive (vozilo/vozila)
# Maps entity ROOT to list of alternative words users might use
#
# COVERAGE CATEGORIES:
# - Fleet & Vehicles (10 groups)
# - People & Roles (8 groups)
# - Bookings & Time (6 groups)
# - Documents & Records (8 groups)
# - Financial (8 groups)
# - Technical & Status (10 groups)
# - Communication (5 groups)
# - Data & Analytics (5 groups)
CROATIAN_SYNONYMS = {
    # ---
    # FLEET & VEHICLES (10 groups)
    # ---
    "vozil": ["auto", "automobil", "kola", "car", "autić"],  # vozilo, vozila
    "flot": ["fleet", "vozni park", "park vozila"],  # flota, flote
    "goriv": ["benzin", "nafta", "dizel", "fuel", "tank", "spremnik"],  # gorivo, goriva
    "kilometraž": ["km", "kilometri", "prijeđeno", "mileage", "stanje sata"],  # kilometraža
    "gum": ["tire", "pneumatik", "kotač"],  # guma, gume
    "baterij": ["akumulator", "battery", "punjenje"],  # baterija, baterije
    "ulj": ["oil", "mazivo", "motor oil"],  # ulje, ulja
    "registracij": ["tablice", "plates", "oznaka", "reg"],  # registracija
    "šasij": ["VIN", "chassis", "broj šasije"],  # šasija, šasije
    "oprema": ["equipment", "dodaci", "accessories"],  # oprema, opreme

    # ---
    # PEOPLE & ROLES (8 groups)
    # ---
    "osob": ["čovjek", "korisnik", "user", "person"],  # osoba, osobe
    "vozač": ["driver", "šofer", "vozar"],  # vozač, vozača
    "kupac": ["customer", "klijent", "mušterija", "stranka"],  # kupac, kupca
    "zaposlenik": ["employee", "radnik", "djelatnik", "worker"],  # zaposlenik
    "korisnik": ["user", "upotrebitelj", "account"],  # korisnik, korisnika
    "tim": ["team", "ekipa", "grupa", "odjel"],  # tim, tima
    "kontakt": ["contact", "osoba", "broj telefona"],  # kontakt, kontakta
    "organizacij": ["company", "tvrtka", "firma", "poduzeće"],  # organizacija

    # ---
    # BOOKINGS & TIME (6 groups)
    # ---
    "rezervacij": ["booking", "najam", "iznajmljivanje", "rent", "narudžba"],  # rezervacija
    "raspored": ["schedule", "plan", "kalendar", "termin"],  # raspored, rasporeda
    "putovanj": ["trip", "vožnja", "ruta", "journey", "put"],  # putovanje
    "dostupnost": ["slobodno", "available", "raspoloživo", "free"],  # dostupnost
    "termin": ["slot", "appointment", "vrijeme", "sat"],  # termin, termina
    "smjen": ["shift", "radno vrijeme", "tura"],  # smjena, smjene

    # ---
    # LOCATIONS (4 groups)
    # ---
    "lokacij": ["mjesto", "adresa", "pozicija", "location", "GPS"],  # lokacija
    "poslovnic": ["branch", "ured", "office", "prodajno mjesto"],  # poslovnica
    "rut": ["route", "put", "pravac", "itinerar"],  # ruta, rute
    "zon": ["zone", "područje", "region", "sektor"],  # zona, zone

    # ---
    # DOCUMENTS & RECORDS (8 groups)
    # ---
    "dokument": ["document", "papir", "spis", "akt"],  # dokument, dokumenta
    "ugovor": ["contract", "dogovor", "sporazum", "agreement"],  # ugovor
    "izvještaj": ["report", "pregled", "statistika", "analiza"],  # izvještaj
    "zapisnik": ["log", "record", "evidencija", "dnevnik"],  # zapisnik, zapisnika
    "povijes": ["history", "prošlost", "arhiva", "staro"],  # povijest, povijesti
    "certifikat": ["certificate", "potvrda", "svjedodžba"],  # certifikat
    "licenc": ["license", "dozvola", "odobrenje"],  # licenca, licence
    "polic": ["policy", "pravilo", "osiguranje"],  # polica, police

    # ---
    # FINANCIAL (8 groups)
    # ---
    "račun": ["faktura", "invoice", "naplata", "bill"],  # račun, računa
    "cijen": ["trošak", "cost", "price", "iznos", "tarifa"],  # cijena, cijene
    "plaćanj": ["uplata", "payment", "transakcija", "plata"],  # plaćanje
    "osiguranj": ["polica", "insurance", "kasko"],  # osiguranje
    "kazn": ["fine", "penalty", "globa", "prekršajna"],  # kazna, kazne
    "naknada": ["fee", "pristojba", "charge", "compensation"],  # naknada
    "depo": ["deposit", "polog", "jamčevina", "avans"],  # depozit, depozita
    "popust": ["discount", "sniženje", "akcija", "sale"],  # popust, popusta

    # ---
    # MAINTENANCE & SERVICE (6 groups)
    # ---
    "održavanj": ["servis", "service", "popravak", "maintenance"],  # održavanje
    "inspekcij": ["inspection", "pregled", "kontrola", "check"],  # inspekcija
    "štet": ["oštećenje", "damage", "kvar", "defekt"],  # šteta, štete
    "nesreć": ["accident", "sudar", "incident", "prometna"],  # nesreća
    "popravak": ["repair", "fix", "servis", "obnova"],  # popravak, popravka
    "kvar": ["malfunction", "defekt", "problem", "neispravnost"],  # kvar, kvara

    # ---
    # COMMUNICATION (5 groups)
    # ---
    "obavijest": ["notification", "notifikacija", "alert", "info"],  # obavijest
    "poruk": ["message", "sms", "email", "pismo"],  # poruka, poruke
    "upozorenje": ["warning", "alert", "alarm", "oprez"],  # upozorenje
    "komentar": ["comment", "napomena", "bilješka", "note"],  # komentar
    "zahtjev": ["request", "molba", "upit", "traženje"],  # zahtjev, zahtjeva

    # ---
    # DATA & ANALYTICS (5 groups)
    # ---
    "podac": ["data", "informacija", "info", "podatak"],  # podaci, podataka
    "statistik": ["statistics", "analitika", "metrics", "brojke"],  # statistika
    "grafikon": ["chart", "graph", "dijagram", "vizualizacija"],  # grafikon
    "pregled": ["overview", "dashboard", "summary", "sažetak"],  # pregled
    "rezultat": ["result", "output", "ishod", "odgovor"],  # rezultat, rezultata

    # ---
    # HANDOVER & TRANSFER (5 groups)
    # ---
    "preuzimanj": ["pickup", "primanje", "dohvat"],  # preuzimanje
    "vraćanj": ["return", "dropoff", "povratak"],  # vraćanje
    "primopredaj": ["handover", "primanje", "predaja"],  # primopredaja
    "prijav": ["checkin", "login", "registracija"],  # prijava, prijave
    "odjav": ["checkout", "logout", "odjavljivanje"],  # odjava, odjave

    # ---
    # CARDS & ACCESS (4 groups)
    # ---
    "kartic": ["card", "ENC", "fuel card", "smartcard"],  # kartica, kartice
    "ključ": ["key", "pristup", "otključavanje"],  # ključ, ključa
    "pristup": ["access", "dozvola", "ulaz"],  # pristup, pristupa
    "uređaj": ["device", "tracker", "GPS", "telematics"],  # uređaj, uređaja

    # ---
    # CATEGORIES & TYPES (3 groups)
    # ---
    "kategorij": ["category", "tip", "vrsta", "klasa"],  # kategorija
    "mark": ["brand", "proizvođač", "marka"],  # marka, marke
    "model": ["model", "verzija", "varijanta"],  # model, modela
}

# Common API prefixes to skip (not meaningful for embedding)
SKIP_SEGMENTS = {
    "api", "v1", "v2", "v3", "v4", "odata", "rest", "public", "private",
    "internal", "external", "admin", "management", "system", "core",
}
