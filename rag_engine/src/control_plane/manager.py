"""
Control Plane Manager - Orchestrates data lifecycle.

Responsibilities:
1. Check if ticker folder exists (full onboarding vs incremental update)
2. Check freshness of each component
3. Fetch stale or missing data
4. Serialize, embed, and upsert to Pinecone
5. Ensure Pinecone is exact mirror of disk

Invariant: Disk is source of truth, Pinecone mirrors it exactly.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from .config import (
    Jurisdiction,
    BASE_DATA_DIR,
    STRUCTURED_COMPONENTS,
    PINECONE_INDEX_NAME,
)
from .company_registry import CompanyInfo, resolve_company
from .freshness import (
    FreshnessResult,
    check_all_freshness,
    ticker_folder_exists,
)


@dataclass
class DataChecklist:
    """Specifies which data components to manage."""
    structured: list[str] = field(default_factory=lambda: STRUCTURED_COMPONENTS.copy())
    unstructured: bool = True

    def get_all_components(self) -> list[str]:
        """Get list of all component names."""
        components = self.structured.copy()
        if self.unstructured:
            components.append("unstructured")
        return components


@dataclass
class ControlPlaneResult:
    """Result from the control plane operations."""
    ticker: str
    jurisdiction: Optional[Jurisdiction]
    folder_existed: bool
    components_checked: dict[str, bool]   # component → was_fresh
    components_updated: list[str]         # components that were refetched
    components_indexed: list[str]         # components that were indexed
    errors: list[str]

    @property
    def success(self) -> bool:
        """True if no critical errors occurred."""
        return len(self.errors) == 0 or len(self.components_indexed) > 0


class ControlPlaneManager:
    """
    Manages data lifecycle for a ticker.

    Ensures Pinecone is exact mirror of disk (disk is source of truth).
    Uses stable vector IDs so upserts overwrite existing vectors.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or BASE_DATA_DIR

    def ensure_data_ready(
        self,
        ticker: str,
        checklist: Optional[DataChecklist] = None,
        cik: Optional[str] = None,
        scrip_code: Optional[str] = None,
        force_refresh: bool = False
    ) -> ControlPlaneResult:
        """
        Main entry point. Ensures all checklist components are fresh and indexed.

        Flow:
        1. Resolve company info (jurisdiction, identifiers)
        2. Check if ticker folder exists
        3. If not exists → full onboarding (fetch all, index all)
        4. If exists → check each component staleness, update stale ones
        5. Return result with what was checked/updated

        Args:
            ticker: Stock ticker symbol
            checklist: Which components to manage (default: all)
            cik: Optional CIK for US companies
            scrip_code: Optional scrip code for Indian companies
            force_refresh: Force refetch regardless of freshness

        Returns:
            ControlPlaneResult with operation details
        """
        ticker = ticker.upper()
        checklist = checklist or DataChecklist()
        errors: list[str] = []
        components_updated: list[str] = []
        components_indexed: list[str] = []
        components_checked: dict[str, bool] = {}

        # Step 1: Resolve company info
        info = resolve_company(ticker, cik, scrip_code)
        if info is None:
            return ControlPlaneResult(
                ticker=ticker,
                jurisdiction=None,
                folder_existed=False,
                components_checked={},
                components_updated=[],
                components_indexed=[],
                errors=[f"Unknown ticker: {ticker}. Provide CIK (US) or scrip_code (India)."]
            )

        # Step 2: Check if folder exists
        folder_existed = ticker_folder_exists(ticker, self.base_dir)

        if not folder_existed or force_refresh:
            # Full onboarding
            print(f"[ControlPlane] {'Force refresh' if force_refresh else 'Full onboarding'} for {ticker}")
            result = self._full_onboarding(ticker, info, checklist)
            components_updated = result["updated"]
            components_indexed = result["indexed"]
            errors.extend(result["errors"])
            # Mark all as not fresh (since we fetched them)
            for comp in checklist.get_all_components():
                components_checked[comp] = False
        else:
            # Incremental update
            print(f"[ControlPlane] Incremental update for {ticker}")
            freshness = check_all_freshness(
                ticker=ticker,
                jurisdiction=info.jurisdiction,
                base_dir=self.base_dir,
                components=checklist.get_all_components()
            )

            for comp, result in freshness.items():
                components_checked[comp] = result.is_fresh
                print(f"  {result}")

            result = self._incremental_update(ticker, info, checklist, freshness)
            components_updated = result["updated"]
            components_indexed = result["indexed"]
            errors.extend(result["errors"])

        return ControlPlaneResult(
            ticker=ticker,
            jurisdiction=info.jurisdiction,
            folder_existed=folder_existed,
            components_checked=components_checked,
            components_updated=components_updated,
            components_indexed=components_indexed,
            errors=errors
        )

    def _full_onboarding(
        self,
        ticker: str,
        info: CompanyInfo,
        checklist: DataChecklist
    ) -> dict:
        """
        Full onboarding: fetch all data, serialize, embed, upsert.

        Returns:
            dict with "updated", "indexed", "errors" lists
        """
        updated: list[str] = []
        indexed: list[str] = []
        errors: list[str] = []

        # Fetch structured data
        for component in checklist.structured:
            try:
                print(f"  Fetching {component}...")
                self._fetch_structured(ticker, component)
                updated.append(component)
            except Exception as e:
                errors.append(f"Error fetching {component}: {str(e)}")

        # Fetch unstructured data
        if checklist.unstructured:
            try:
                print(f"  Fetching unstructured data...")
                self._fetch_unstructured(ticker, info)
                updated.append("unstructured")
            except Exception as e:
                errors.append(f"Error fetching unstructured: {str(e)}")

        # Index all data to Pinecone
        if updated:
            try:
                print(f"  Indexing to Pinecone...")
                self._index_all(ticker)
                indexed = updated.copy()
            except Exception as e:
                errors.append(f"Error indexing: {str(e)}")

        return {"updated": updated, "indexed": indexed, "errors": errors}

    def _incremental_update(
        self,
        ticker: str,
        info: CompanyInfo,
        checklist: DataChecklist,
        freshness: dict[str, FreshnessResult]
    ) -> dict:
        """
        Incremental update: only fetch and index stale components.

        Uses same vector IDs to overwrite existing vectors in Pinecone.

        Returns:
            dict with "updated", "indexed", "errors" lists
        """
        updated: list[str] = []
        indexed: list[str] = []
        errors: list[str] = []

        # Process structured components
        for component in checklist.structured:
            result = freshness.get(component)
            if result and (not result.exists or not result.is_fresh):
                try:
                    print(f"  Fetching stale {component}...")
                    self._fetch_structured(ticker, component)
                    updated.append(component)
                except Exception as e:
                    errors.append(f"Error fetching {component}: {str(e)}")

        # Process unstructured
        if checklist.unstructured:
            result = freshness.get("unstructured")
            if result and (not result.exists or not result.is_fresh):
                try:
                    print(f"  Fetching stale unstructured data...")
                    self._fetch_unstructured(ticker, info)
                    updated.append("unstructured")
                except Exception as e:
                    errors.append(f"Error fetching unstructured: {str(e)}")

        # Index updated components
        if updated:
            try:
                print(f"  Indexing updated components to Pinecone...")
                # For incremental updates, we re-index all to ensure consistency
                # The stable IDs ensure overwrites, not duplicates
                self._index_all(ticker)
                indexed = updated.copy()
            except Exception as e:
                errors.append(f"Error indexing: {str(e)}")

        return {"updated": updated, "indexed": indexed, "errors": errors}

    def _fetch_structured(self, ticker: str, component: str) -> None:
        """Fetch structured data via yfinance and serialize to JSON."""
        from src.structured.data import fetch_and_store_stock_data
        fetch_and_store_stock_data(ticker, component)

    def _fetch_unstructured(self, ticker: str, info: CompanyInfo) -> None:
        """Fetch unstructured data (SEC 10-K for US, BSE filings for India)."""
        if info.jurisdiction == Jurisdiction.US:
            if not info.cik:
                raise ValueError(f"CIK required for US company {ticker}")
            from src.unstructured_data.ingestion_unstructured_foreign import ingest_sec_unstructured
            ingest_sec_unstructured(ticker=ticker, cik=info.cik)
        else:
            if not info.scrip_code:
                raise ValueError(f"Scrip code required for Indian company {ticker}")
            from src.unstructured_data.ingestion_unstructured_indian import process_company
            process_company(ticker, info.scrip_code)

    def _index_all(self, ticker: str) -> None:
        """Index all data for a ticker to Pinecone."""
        from src.indexing.upsert_pinecone import index_all_data
        index_all_data(ticker)
        
        # Trigger Semantic Cache Invalidation
        try:
            from retrieval.semantic_cache import clear_cache
            clear_cache()
            print("[ControlPlane] Semantic cache cleared.")
        except ImportError:
            pass # Ignore if cache module not available
        except Exception as e:
            print(f"[ControlPlane] Warning: Failed to clear semantic cache: {e}")

    def _index_component(self, ticker: str, component: str) -> None:
        """
        Index a single component to Pinecone.

        Uses stable vector IDs so this overwrites existing vectors.
        """
        # For now, delegate to index_all. In future, could add
        # component-specific indexing for better efficiency.
        self._index_all(ticker)
