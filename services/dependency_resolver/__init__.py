"""
Dependency Resolver Package

Re-exports for backward compatibility:
    from services.dependency_resolver import DependencyResolver, ResolutionResult, EntityReference
"""

from services.dependency_resolver.models import ResolutionResult, EntityReference
from services.dependency_resolver.resolver import DependencyResolver

__all__ = [
    "DependencyResolver",
    "ResolutionResult",
    "EntityReference",
]
