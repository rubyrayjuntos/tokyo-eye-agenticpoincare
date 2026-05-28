"""
Validation Report Service

Fetches and parses wwPDB validation report XML into structured outlier data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gzip
from typing import List, Optional
import xml.etree.ElementTree as ET

import requests

from gosp.models.data_models import BondOutlier, Clash


@dataclass
class ValidationReport:
    """Parsed validation report content."""
    pdb_id: str
    bond_outliers: List[BondOutlier]
    angle_outliers: List[BondOutlier]
    clashes: List[Clash]


class ValidationReportError(Exception):
    """Raised when validation report fetching or parsing fails."""
    pass


class ValidationReportParser:
    """Parses wwPDB validation reports (XML format)."""

    BASE_URL = "https://files.wwpdb.org/pub/pdb/validation_reports"

    def __init__(self, pdb_id: str, cache_dir: Optional[Path] = None):
        self.pdb_id = pdb_id.upper()
        self.cache_dir = cache_dir

    def fetch_xml(self) -> str:
        """Download validation report XML (cached if available)."""
        cache_path = None
        if self.cache_dir:
            cache_path = self.cache_dir / "validation" / f"{self.pdb_id.lower()}_validation.xml"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.exists():
                return cache_path.read_text(encoding="utf-8")

        subdir = self.pdb_id[1:3].lower()
        pdb_lower = self.pdb_id.lower()
        base_url = f"{self.BASE_URL}/{subdir}/{pdb_lower}/{pdb_lower}_validation"
        url = f"{base_url}.xml"
        response = requests.get(url, timeout=30)
        if response.ok:
            xml_text = response.text
        else:
            gz_url = f"{base_url}.xml.gz"
            gz_response = requests.get(gz_url, timeout=30)
            if not gz_response.ok:
                raise ValidationReportError(
                    f"Validation report not found for {self.pdb_id} (status {gz_response.status_code})"
                )
            try:
                xml_text = gzip.decompress(gz_response.content).decode("utf-8")
            except OSError as exc:
                raise ValidationReportError(
                    f"Failed to decompress validation report for {self.pdb_id}: {exc}"
                )
        if cache_path:
            cache_path.write_text(xml_text, encoding="utf-8")
        return xml_text

    def parse(self, xml_content: str) -> ET.Element:
        """Parse XML string into ElementTree."""
        try:
            return ET.fromstring(xml_content)
        except ET.ParseError as exc:
            raise ValidationReportError(f"Failed to parse validation XML: {exc}")

    def extract_bond_outliers(self, root: ET.Element) -> List[BondOutlier]:
        """Extract bond length outliers."""
        outliers: List[BondOutlier] = []
        for elem in root.findall(".//bond-outlier"):
            atoms_str = (elem.findtext("atoms") or "").strip()
            if "-" not in atoms_str:
                continue
            atoms = tuple(atoms_str.split("-", 1))
            outliers.append(
                BondOutlier(
                    mol_id=int(elem.findtext("mol-id") or 0),
                    chain=(elem.findtext("chain") or ""),
                    residue_id=int(elem.findtext("residue") or 0),
                    residue_type=(elem.findtext("type") or ""),
                    atoms=(atoms[0], atoms[1]),
                    z_score=float(elem.findtext("z-score") or 0.0),
                    observed=float(elem.findtext("observed") or 0.0),
                    ideal=float(elem.findtext("ideal") or 0.0),
                    deviation=float(elem.findtext("observed") or 0.0)
                    - float(elem.findtext("ideal") or 0.0),
                    outlier_type="length",
                )
            )
        return outliers

    def extract_angle_outliers(self, root: ET.Element) -> List[BondOutlier]:
        """Extract bond angle outliers."""
        outliers: List[BondOutlier] = []
        for elem in root.findall(".//angle-outlier"):
            atoms_str = (elem.findtext("atoms") or "").strip()
            if "-" not in atoms_str:
                continue
            atoms = tuple(atoms_str.split("-", 1))
            outliers.append(
                BondOutlier(
                    mol_id=int(elem.findtext("mol-id") or 0),
                    chain=(elem.findtext("chain") or ""),
                    residue_id=int(elem.findtext("residue") or 0),
                    residue_type=(elem.findtext("type") or ""),
                    atoms=(atoms[0], atoms[1]),
                    z_score=float(elem.findtext("z-score") or 0.0),
                    observed=float(elem.findtext("observed") or 0.0),
                    ideal=float(elem.findtext("ideal") or 0.0),
                    deviation=float(elem.findtext("observed") or 0.0)
                    - float(elem.findtext("ideal") or 0.0),
                    outlier_type="angle",
                )
            )
        return outliers

    def extract_clashes(self, root: ET.Element) -> List[Clash]:
        """Extract clash entries."""
        clashes: List[Clash] = []
        for elem in root.findall(".//clash"):
            clashes.append(
                Clash(
                    atom1=(elem.findtext("atom1") or ""),
                    atom2=(elem.findtext("atom2") or ""),
                    distance=float(elem.findtext("distance") or 0.0),
                    clash_magnitude=float(elem.findtext("magnitude") or 0.0),
                )
            )
        return clashes

    def fetch_and_parse(self) -> ValidationReport:
        """Fetch and parse validation report."""
        xml_text = self.fetch_xml()
        root = self.parse(xml_text)
        bond_outliers = self.extract_bond_outliers(root)
        angle_outliers = self.extract_angle_outliers(root)
        clashes = self.extract_clashes(root)
        return ValidationReport(
            pdb_id=self.pdb_id,
            bond_outliers=bond_outliers,
            angle_outliers=angle_outliers,
            clashes=clashes,
        )
