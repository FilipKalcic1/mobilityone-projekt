"""
User Handler - User identification and greeting.
Version: 1.0

Extracted from engine/__init__.py for better modularity.
"""

import logging
from typing import Dict, Any, Optional, Tuple

from services.user_service import UserService
from services.context import UserContextManager

logger = logging.getLogger(__name__)


class UserHandler:
    """
    Handles user identification and greeting.

    Responsibilities:
    - Identify user from phone number
    - Auto-onboard new users
    - Build personalized greetings
    """

    def __init__(self, db_session, gateway, cache_service):
        """
        Initialize UserHandler.

        Args:
            db_session: Database session
            gateway: API gateway
            cache_service: Cache service
        """
        self.db = db_session
        self.gateway = gateway
        self.cache = cache_service

    async def identify_user(self, phone: str, db_session=None) -> Optional[Dict[str, Any]]:
        """
        Identify user and build context with dynamic tenant resolution.

        ALWAYS returns a context - never None.
        If user is not in MobilityOne, returns guest context so bot still works.

        Args:
            phone: User phone number
            db_session: Database session for this request (required for concurrency safety)

        Returns:
            User context dict (always non-None)
        """
        db = db_session or self.db
        user_service = UserService(db, self.gateway, self.cache)

        user = await user_service.get_active_identity(phone)

        if user:
            # Pass user_mapping for dynamic tenant resolution
            ctx = await user_service.build_context(user.api_identity, phone, user_mapping=user)
            ctx["display_name"] = user.display_name
            ctx["is_new"] = False
            return ctx

        result = await user_service.try_auto_onboard(phone)

        if result:
            display_name, vehicle_data = result
            user = await user_service.get_active_identity(phone)

            if user:
                # Pass user_mapping for dynamic tenant resolution
                ctx = await user_service.build_context(user.api_identity, phone, user_mapping=user)
                ctx["display_name"] = display_name
                ctx["is_new"] = True
                return ctx

        # User not found in MobilityOne - return guest context
        # Bot still works, just without vehicle-specific features
        logger.info(f"Guest user: {phone[-4:]}... - not in MobilityOne, creating guest context")
        return {
            "person_id": None,
            "phone": phone,
            "tenant_id": user_service.default_tenant_id,
            "display_name": "Korisnik",
            "vehicle": {},
            "is_new": True,
            "is_guest": True
        }

    def build_greeting(self, user_context: Dict[str, Any]) -> str:
        """
        Build personalized greeting for new user.

        Args:
            user_context: User context dict

        Returns:
            Greeting message
        """
        # Guest user greeting
        if user_context.get("is_guest"):
            return (
                "Pozdrav!\n\n"
                "Ja sam MobilityOne AI asistent.\n\n"
                "Vas broj nije registriran u sustavu, ali svejedno vam mogu pomoci "
                "s opcim informacijama.\n\n"
                "Kako vam mogu pomoci?"
            )

        ctx = UserContextManager(user_context)
        vehicle = ctx.vehicle

        greeting = f"Pozdrav {ctx.display_name}!\n\n"
        greeting += "Ja sam MobilityOne AI asistent.\n\n"

        if vehicle and vehicle.plate:
            greeting += f"Vidim da vam je dodijeljeno vozilo:\n"
            greeting += f"   **{vehicle.name or 'vozilo'}** ({vehicle.plate})\n"
            greeting += f"   Kilometraza: {vehicle.mileage or 'N/A'} km\n\n"
            greeting += "Kako vam mogu pomoci?\n"
            greeting += "* Unos kilometraze\n"
            greeting += "* Prijava kvara\n"
            greeting += "* Rezervacija vozila\n"
            greeting += "* Pitanja o vozilu"
        elif vehicle and vehicle.id:
            greeting += f"Vidim da vam je dodijeljeno vozilo: {vehicle.name or 'vozilo'}\n\n"
            greeting += "Kako vam mogu pomoci?"
        else:
            greeting += "Trenutno nemate dodijeljeno vozilo.\n\n"
            greeting += "Zelite li rezervirati vozilo? Recite mi:\n"
            greeting += "* Za koji period (npr. 'sutra od 8 do 17')\n"
            greeting += "* Ili samo recite 'Trebam vozilo' pa cemo dalje"

        return greeting
