import pandas as pd
from typing import Dict

class HydrogenTableAnalyzer:
    def __init__(self, data_file: str = 'data/hydrogentable.xlsx'):
        """Initialize the Hydrogen Table Analyzer with clean data structure."""
        self.data_file = data_file
        self.mapping = None
        self.shock_data = None
        self.coefficients = {}  # Will store productioncoeff, valueaddedcoeff, jobcoeff, directemploycoeff
        self.load_data()

    def load_data(self):
        """Load mapping and all coefficient matrices from Excel file."""
        print("Loading Hydrogen Table data...")

        # Load mapping sheet
        self.mapping = pd.read_excel(self.data_file, sheet_name='basicmap')
        print(f"Loaded {len(self.mapping)} sectors from basicmap sheet")

        # # Load SHOCK data for additional context
        # self.shock_data = pd.read_excel(self.data_file, sheet_name='SHOCK')
        # print(f"Loaded SHOCK data with {len(self.shock_data)} scenarios")

        # Load production coefficient
        df_A = pd.read_excel(self.data_file, sheet_name='productioncoeff')
        self.coefficients['productioncoeff'] = df_A.set_index('code')
        print(f"Loaded production-inducing coefficient matrix: {self.coefficients['productioncoeff'].shape}")

        # Load value-added coefficients
        df_value_added = pd.read_excel(self.data_file, sheet_name='valueaddedcoeff')
        self.coefficients['valueaddedcoeff'] = df_value_added.set_index('code')
        print(f"Loaded value-added coefficient matrix: {self.coefficients['valueaddedcoeff'].shape}")

        # Load job coefficients (total job creation) 임금유발효과
        df_jobcoeff = pd.read_excel(self.data_file, sheet_name='jobcoeff')
        self.coefficients['jobcoeff'] = df_jobcoeff.set_index('code')
        print(f"Loaded wage-inducing coefficient matrix: {self.coefficients['jobcoeff'].shape}")

        # Load direct employment coefficients 취업유발효과
        df_directemploy = pd.read_excel(self.data_file, sheet_name='directemploycoeff')
        self.coefficients['directemploycoeff'] = df_directemploy.set_index('code')
        print(f"Loaded job creation coefficient matrix: {self.coefficients['directemploycoeff'].shape}")

        # Create code-to-product mapping dictionary
        self.code_to_product = {}
        self.code_to_product_display = {}  # For display purposes

        for _, row in self.mapping.iterrows():
            code = row['code']
            product = row['product']

            self.code_to_product[code] = product
            # Store display version showing the code and product name
            self.code_to_product_display[code] = f"{code}: {product}"

        print("Hydrogen Table data loading complete!")

    def get_sector_options(self) -> Dict:
        """Return all available sector codes and their products for display."""
        return self.code_to_product_display

    def get_sector_from_display(self, display_string: str):
        """Extract the sector code from display string."""
        code_part = display_string.split(":")[0]
        return code_part

    def get_hydrogen_scenarios(self) -> list:
        """Return available hydrogen scenarios (H2P, H2S, H2T, H2U)."""
        return ['H2P', 'H2S', 'H2T', 'H2U']

    def calculate_hydrogen_effects(self, scenario: str, demand_change: float, coeff_type: str = 'productioncoeff', quiet: bool = False) -> Dict[str, any]:
        """
        Calculate effects of hydrogen scenario using specified coefficient matrix.

        Args:
            scenario: Hydrogen scenario (H2P, H2S, H2T, H2U)
            demand_change: Change in final demand (positive or negative)
            coeff_type: Type of coefficients to use
            quiet: If True, suppress print output

        Returns:
            Dictionary with analysis results
        """
        if scenario not in self.get_hydrogen_scenarios():
            raise ValueError(f"Scenario {scenario} not found. Available scenarios: {self.get_hydrogen_scenarios()}")

        if coeff_type not in self.coefficients:
            raise ValueError(f"Coefficient type '{coeff_type}' not available. Choose from: {list(self.coefficients.keys())}")

        coeff_names = {
            'productioncoeff': 'Production-inducing Coefficients',
            'valueaddedcoeff': 'Value-Added Coefficients',
            'jobcoeff': 'Wage-inducing Coefficients',
            'directemploycoeff': 'Total job creation Coefficients'
        }

        if not quiet:
            print(f"\nAnalyzing {coeff_names[coeff_type]} effects for hydrogen scenario: {scenario}")
            print(f"Demand change: {demand_change:,.0f}")
            print(f"Using coefficient type: {coeff_type} ({coeff_names[coeff_type]})")

        # Get coefficient matrix
        selected_coeffs = self.coefficients[coeff_type]

        if scenario not in selected_coeffs.columns:
            raise ValueError(f"Column for scenario {scenario} not found in {coeff_type} coefficient matrix")

        # Calculate direct effects: coefficient * demand_change
        if coeff_type == "directemploycoeff":
            # For employment: (person/billion won) * (million won / 1000) = persons
            direct_impacts = selected_coeffs[scenario] * demand_change/1000
        else:
            # For economic effects: convert from million won to billion won
            direct_impacts = selected_coeffs[scenario] * demand_change / 1000

        # Remove zero or near-zero impacts and NaN values
        #significant_impacts = direct_impacts[(abs(direct_impacts) > 1e-6) & pd.notna(direct_impacts)]
        significant_impacts = direct_impacts
        # Create results with sector names
        results = []
        for sector_code, impact in significant_impacts.items():
            if sector_code in self.code_to_product:
                results.append({
                    'sector_code': sector_code,
                    'sector_name': self.code_to_product[sector_code],
                    'impact': impact
                })

        # Sort by absolute impact (descending)
        results.sort(key=lambda x: abs(x['impact']), reverse=True)

        # Calculate summary statistics

        total_impact = sum([r['impact'] for r in results])

        return {
            'scenario': scenario,
            'demand_change': demand_change,
            'coeff_type': coeff_type,
            'coeff_name': coeff_names[coeff_type],
            'impacts': results,
            'total_impact': total_impact,
            'num_affected_sectors': len(results)
        }

    def display_results(self, results: Dict):
        """Display analysis results in a formatted way."""
        print(f"\n{'='*60}")
        print(f"HYDROGEN EFFECTS ANALYSIS - {results['coeff_name'].upper()}")
        print(f"{'='*60}")
        print(f"Hydrogen Scenario: {results['scenario']}")
        print(f"Demand Change (million won): {results['demand_change']:,.0f}")
        print(f"Coefficient Type: {results['coeff_type']} ({results['coeff_name']})")
        print(f"Total Economic Impact (million won): {results['impacts'] == 'productioncoeff':,.2f}")
        print(f"Total Job Impact (person/billion won): {results['total_impact']:,.2f}")
        print(f"Affected Sectors: {results['num_affected_sectors']}")

        print(f"\n{'Top 20 Impacts:':<60}")
        print(f"{'Code':<6} {'Sector':<35} {'Impact':>15}")
        print("-" * 60)

        for impact in results['impacts'][:20]:
            print(f"{impact['sector_code']:<6} {impact['sector_name']:<35} {impact['impact']:>15,.2f}")

        if len(results['impacts']) > 20:
            print(f"\n... and {len(results['impacts']) - 20} more sectors with smaller impacts")

    def calculate_all_effects(self, scenario: str, demand_change: float, quiet: bool = True) -> tuple:
        """
        Calculate all coefficient effects for a given hydrogen scenario and demand change.

        Args:
            scenario: Hydrogen scenario (H2P, H2S, H2T, H2U)
            demand_change: Change in final demand (positive or negative)
            quiet: If True, suppress print output (default True for GUI use)

        Returns:
            Tuple of (all_results, coefficient_types, coeff_names)
            - all_results: Dictionary with results for each coefficient type
            - coefficient_types: List of coefficient types analyzed
            - coeff_names: Dictionary mapping coefficient types to display names
        """
        # Define coefficient types and their display names
        coefficient_types = ["productioncoeff", "valueaddedcoeff", "jobcoeff", "directemploycoeff"]
        coeff_names = {
            "productioncoeff": "Production-inducing",
            "valueaddedcoeff": "Value-Added",
            "jobcoeff": "Wage-inducing",
            "directemploycoeff": "Total Job Creation"
        }

        # Calculate all coefficient effects
        all_results = {}
        for coeff_type in coefficient_types:
            try:
                results = self.calculate_hydrogen_effects(
                    scenario,
                    demand_change,
                    coeff_type,
                    quiet=quiet
                )
                all_results[coeff_type] = results
            except Exception as e:
                if not quiet:
                    print(f"Error calculating {coeff_type}: {str(e)}")
                all_results[coeff_type] = None

        return all_results, coefficient_types, coeff_names

    def create_combined_data(self, all_results: Dict, coefficient_types: list, coeff_names: Dict) -> list:
        """
        Create combined data from all analysis results.

        Args:
            all_results: Dictionary of results from calculate_hydrogen_effects for each coefficient type
            coefficient_types: List of coefficient types to include
            coeff_names: Dictionary mapping coefficient types to display names

        Returns:
            List of dictionaries containing combined data for all coefficient types
        """
        combined_data = []

        for coeff_type in coefficient_types:
            if all_results[coeff_type] and all_results[coeff_type]['impacts']:
                results = all_results[coeff_type]
                for impact in results['impacts']:
                    sector_code = impact['sector_code']

                    combined_data.append({
                        'coefficient_type': coeff_type,
                        'coefficient_name': coeff_names[coeff_type],
                        'sector_code': sector_code,
                        'sector_name': impact['sector_name'],
                        'impact': impact['impact']
                    })

        return combined_data

    # def get_shock_data(self) -> pd.DataFrame:
    #     """Return the SHOCK data for reference."""
    #     return self.shock_data