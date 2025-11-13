import pandas as pd
from typing import Dict

class IOTableAnalyzer:
    def __init__(self, data_file: str = 'data/iotable_2023.xlsx'):
        """Initialize the I-O Table Analyzer with clean data structure."""
        self.data_file = data_file
        self.mapping = None
        self.codemap = None
        self.subsectormap = None
        self.coefficients = {}  # Will store A, Am, Ad, job coefficients
        self.basic_to_subsector = {}  # Mapping from basic sector to sub-sector
        self.basic_to_subsector_to_code_h = {} #added
        self.subsector_to_name = {}  # Mapping from sub-sector code to name
        self.load_data()
    
    def load_data(self):
        """Load mapping and all three coefficient matrices from Excel file."""
        print("Loading I-O Table data...")

        def format_code(code):
            code_str = str(code).strip()
            if code_str.isdigit() and len(code_str) == 3:
                return f"0{code_str}"
            return code_str

        def format_subsector_code(code):
            try:
                code_int = int(float(code))
                return f"{code_int:03d}"
            except (ValueError, TypeError):
                return str(code).strip()
        
        # Load mapping sheet
        self.mapping = pd.read_excel(self.data_file, sheet_name='basicmap')
        self.mapping['code'] = self.mapping['code'].apply(format_code)
        print(f"Loaded {len(self.mapping)} sectors from basicmap sheet")
        
        # # Load direct input coefficients (A)
        # df_A = pd.read_excel(self.data_file, sheet_name='directinputcoeff_A')
        # self.coefficients['A'] = df_A.set_index('code')
        # print(f"Loaded A (direct) coefficient matrix: {self.coefficients['A'].shape}")
        
        # # Load import input coefficients (Am)
        # df_Am = pd.read_excel(self.data_file, sheet_name='importinputcoeff_Am')
        # self.coefficients['Am'] = df_Am.set_index('code')
        # print(f"Loaded Am (import) coefficient matrix: {self.coefficients['Am'].shape}")
        
        # # Load domestic coefficients (Ad)
        # df_Ad = pd.read_excel(self.data_file, sheet_name='domesticinputcoeff_Ad')
        # # Clean the column name if needed
        # if 'code' not in df_Ad.columns:
        #     df_Ad = df_Ad.rename(columns={df_Ad.columns[0]: 'code'})
        # self.coefficients['Ad'] = df_Ad.set_index('code')
        # print(f"Loaded Ad (domestic) coefficient matrix: {self.coefficients['Ad'].shape}")
        
        # Load indirect production coefficients (I-Ad)^-1
        df_indirect_prod = pd.read_excel(self.data_file, sheet_name='indirectprodcoeff')
        self.coefficients['indirect_prod'] = df_indirect_prod.set_index('code')
        print(f"Loaded indirect production coefficient matrix: {self.coefficients['indirect_prod'].shape}")
        
        # Load indirect import coefficients
        df_indirect_import = pd.read_excel(self.data_file, sheet_name='indirectimportcoeff')
        self.coefficients['indirect_import'] = df_indirect_import.set_index('code')
        print(f"Loaded indirect import coefficient matrix: {self.coefficients['indirect_import'].shape}")
        
        # Load value-added coefficients
        df_value_added = pd.read_excel(self.data_file, sheet_name='valueaddedcoeff')
        self.coefficients['value_added'] = df_value_added.set_index('code')
        print(f"Loaded value-added coefficient matrix: {self.coefficients['value_added'].shape}")
        
        # Load job coefficients (total job creation)
        df_jobcoeff = pd.read_excel(self.data_file, sheet_name='jobcoeff')
        self.coefficients['jobcoeff'] = df_jobcoeff.set_index('code')
        print(f"Loaded job coefficient matrix: {self.coefficients['jobcoeff'].shape}")
        
        # Load direct employment coefficients
        df_directemploy = pd.read_excel(self.data_file, sheet_name='directemploycoeff')
        self.coefficients['directemploycoeff'] = df_directemploy.set_index('code')
        print(f"Loaded direct employment coefficient matrix: {self.coefficients['directemploycoeff'].shape}")
        
        basic_coeff_sheets = {
            'indirect_prod': 'indirectprodcoeff',
            'indirect_import': 'indirectimportcoeff',
            'value_added': 'valueaddedcoeff'
        }

        for name, sheet in basic_coeff_sheets.items():
            df = pd.read_excel(self.data_file, sheet_name=sheet)
            
            # (수정 포인트 3) 첫 번째 열(인덱스)의 형식을 통일
            df = df.rename(columns={df.columns[0]: 'code'})
            df['code'] = df['code'].apply(format_code)
            df = df.set_index('code')
            
            # (수정 포인트 4) 나머지 모든 열(컬럼)의 형식도 통일
            df.columns = [format_code(col) for col in df.columns]
            
            self.coefficients[name] = df
            print(f"Loaded {name} coefficient matrix: {df.shape} (formatted)")

        job_coeff_sheets = {
            'jobcoeff': 'jobcoeff',
            'directemploycoeff': 'directemploycoeff'
        }
        for name, sheet in job_coeff_sheets.items():
            df = pd.read_excel(self.data_file, sheet_name=sheet)
            df = df.rename(columns={df.columns[0]: 'code'})
            # 인덱스와 컬럼 모두 3자리 소분류 코드 형식으로 통일
            df['code'] = df['code'].apply(format_subsector_code)
            df = df.set_index('code')
            df.columns = [format_subsector_code(col) for col in df.columns]
            self.coefficients[name] = df
            print(f"Loaded {name} coefficient matrix: {df.shape} (sub-sector formatted)")

        # Load codemap for basic-to-subsector mapping
        self.codemap = pd.read_excel(self.data_file, sheet_name='codemap')
        print(f"Loaded codemap with {len(self.codemap)} sector mappings")
        
        # Load subsectormap for sub-sector names
        self.subsectormap = pd.read_excel(self.data_file, sheet_name='subsectormap')
        print(f"Loaded subsectormap with {len(self.subsectormap)} sub-sector names")

        # Create sub-sector code to name mapping
        for _, row in self.subsectormap.iterrows():
            subsector_code = format_subsector_code(row['code'])
            subsector_name = row['name']
            self.subsector_to_name[subsector_code] = subsector_name
        
        print(f"Created sub-sector-to-name mapping for {len(self.subsector_to_name)} sub-sectors")
        
        # Create code_h to product_h mapping
        self.code_h_to_product_h = {}
        self.code_h_to_product_h = pd.Series(self.codemap['product_h'].values, index=self.codemap['code_h']).to_dict()
        print(f"Created code_h_to_product_h mapping with {len(self.code_h_to_product_h)} entries.")

        # Create basic code to code_h mapping
        self.basic_to_code_h = {}
        for _, row in self.codemap.iterrows():
            basic_code = format_code(row['Basic'])
            code_h_value = row['code_h']
            self.basic_to_code_h[basic_code] = code_h_value
        print(f"Created basic_to_code_h mapping with {len(self.basic_to_code_h)} entries.")

        # Create display options for code_h
        unique_code_h = self.codemap[['code_h', 'product_h']].drop_duplicates()
        self.code_h_options = {row['code_h']: f"{row['code_h']}: {row['product_h']}"
                               for _, row in unique_code_h.iterrows()}
        print(f"Created code_h display options with {len(self.code_h_options)} entries.")

        # Create basic-to-subsector mapping
        for _, row in self.codemap.iterrows():
            basic_code = format_code(row['Basic']) # 4자리 형식 적용
            subsector_code = format_subsector_code(row['Sub-sector']) # 3자리 형식 적용
            code_h_value = row['code_h']
            self.basic_to_subsector[basic_code] = subsector_code
            self.basic_to_subsector_to_code_h.setdefault(basic_code, {})[subsector_code] = code_h_value
        
        print(f"Created basic-to-subsector mapping for {len(self.basic_to_subsector)} sectors")
        print(f"Created basic-to-subsector-to-code_h mapping for {len(self.basic_to_subsector_to_code_h)} basic sectors")

        # Create code-to-product mapping dictionary with proper string formatting
        self.code_to_product = {}
        self.code_to_product_display = {}  # For display purposes
        
        # Use first coefficient matrix to check column format
        sample_coeffs = self.coefficients['indirect_prod']
        
        self.code_to_product = pd.Series(self.mapping['product'].values, index=self.mapping['code']).to_dict()
        self.code_to_product_display = {code: f"{code}: {product}" for code, product in self.code_to_product.items()}
        print(f"Created code_to_product mapping with {len(self.code_to_product)} entries.")

        print(f"Final code_to_product mapping size: {len(self.code_to_product)}")
        if len(self.code_to_product) != 380 and len(self.code_to_product_display) != 411:
            print("WARNING: Mapping dictionary size is not 380!")
            # 매핑에 없는 부문 코드 찾기 (고급 디버깅)
            map_codes = set(self.mapping['code'].astype(str))
            mapped_keys = set(str(k) for k in self.code_to_product.keys())
            missing_in_map = map_codes - mapped_keys
            print(f"Codes in basicmap but missing in final mapping: {missing_in_map}")

        #print("Data loading complete!")
    
    def get_sector_options(self, level: str = 'basic') -> Dict:
        """
        Return available sector codes and their products for display.

        Args:
            level: 'basic' for basic sector codes, 'code_h' for high-level categories

        Returns:
            Dictionary mapping codes to display strings
        """
        if level == 'code_h':
            return self.code_h_options
        else:
            return self.code_to_product_display
    
    def get_sector_from_display(self, display_string: str):
        return display_string.split(":")[0]
    
    def calculate_direct_effects(self, target_sector, demand_change: float, coeff_type: str = 'indirect_prod', quiet: bool = False) -> Dict[str, any]:
        """
        Calculate effects of demand change in target sector using specified coefficient matrix.
        
        Args:
            target_sector: Sector code (string like "0111" or integer like 2711)
            demand_change: Change in final demand (positive or negative)
            coeff_type: Type of coefficients to use
            quiet: If True, suppress print output
            
        Returns:
            Dictionary with analysis results
        """
        # Convert target_sector to the proper format used internally
        if isinstance(target_sector, str) and target_sector.isdigit():
            target_sector_int = int(target_sector)
        elif isinstance(target_sector, int):
            target_sector_int = target_sector
        else:
            target_sector_int = None
            
        # Check both string and integer formats
        if target_sector not in self.code_to_product and target_sector_int not in self.code_to_product:
            raise ValueError(f"Sector {target_sector} not found in data")
            
        # Use the format that exists in the mapping
        if target_sector in self.code_to_product:
            final_target_sector = target_sector
        else:
            final_target_sector = target_sector_int
        
        if coeff_type not in self.coefficients:
            raise ValueError(f"Coefficient type '{coeff_type}' not available. Choose from: {list(self.coefficients.keys())}")
        
        target_product = self.code_to_product[final_target_sector]
        coeff_names = {
           # 'A': 'Direct Total', 
            #'Am': 'Direct Import', 
            #'Ad': 'Direct Domestic',
            'indirect_prod': 'Indirect Production (I-Ad)⁻¹',
            'indirect_import': 'Indirect Import',
            'value_added': 'Value-Added',
            'jobcoeff': 'Total Job Creation',
            'directemploycoeff': 'Direct Employment'
        }
        
        if not quiet:
            print(f"\nAnalyzing {coeff_names[coeff_type]} effects for {final_target_sector}: {target_product}")
            print(f"Demand change: {demand_change:,.0f}")
            print(f"Using coefficient type: {coeff_type} ({coeff_names[coeff_type]})")
        
        # Handle job coefficients which use sub-sector mapping
        if coeff_type in ['jobcoeff', 'directemploycoeff']:
            return self._calculate_job_effects(final_target_sector, demand_change, coeff_type, coeff_names[coeff_type], quiet)
        
        # Use final_target_sector for regular coefficients  
        selected_coeffs = self.coefficients[coeff_type]
        
        if final_target_sector not in selected_coeffs.columns:
            raise ValueError(f"Column for sector {final_target_sector} not found in {coeff_type} coefficient matrix")
        
        # Calculate direct effects: coefficient * demand_change
        direct_impacts = selected_coeffs[final_target_sector] * demand_change
        
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
            'target_sector': final_target_sector,
            'target_product': target_product,
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
        print(f"DIRECT EFFECTS ANALYSIS - {results['coeff_name'].upper()}")
        print(f"{'='*60}")
        print(f"Target Sector: {results['target_sector']} - {results['target_product']}")
        print(f"Demand Change: {results['demand_change']:,.0f}")
        print(f"Coefficient Type: {results['coeff_type']} ({results['coeff_name']})")
        print(f"Total Impact: {results['total_impact']:,.2f}")
        print(f"Affected Sectors: {results['num_affected_sectors']}")
        
        print(f"\n{'Top 20 Impacts:':<60}")
        print(f"{'Code':<6} {'Sector':<35} {'Impact':>15}")
        print("-" * 60)
        
        for impact in results['impacts'][:20]:
            print(f"{impact['sector_code']:<6} {impact['sector_name']:<35} {impact['impact']:>15,.2f}")
        
        if len(results['impacts']) > 20:
            print(f"\n... and {len(results['impacts']) - 20} more sectors with smaller impacts")
    
    def _calculate_job_effects(self, target_sector, demand_change: float, coeff_type: str, coeff_name: str, quiet: bool = False) -> Dict[str, any]:
        """
        Calculate job effects using sub-sector mapping.
        Job coefficients use sub-sector codes, so we need to map basic sector to sub-sector first.
        """
        target_product = self.code_to_product[target_sector]
        
        # Find the sub-sector code for this basic sector
        if target_sector not in self.basic_to_subsector:
            raise ValueError(f"Sub-sector mapping not found for basic sector {target_sector}")
        
        subsector_code = self.basic_to_subsector[target_sector]
        
        if not quiet:
            print(f"Basic sector {target_sector} maps to sub-sector {subsector_code}")
        
        # Get job coefficient matrix
        selected_coeffs = self.coefficients[coeff_type]
        
        # Check if sub-sector column exists in job coefficient matrix
        if subsector_code not in selected_coeffs.columns:
            raise ValueError(f"Sub-sector column {subsector_code} not found in {coeff_type} coefficient matrix")
        
        # Calculate job effects: coefficient * demand_change
        # Note: Job coefficients represent jobs per unit of output, so result is in number of jobs
        job_impacts = selected_coeffs[subsector_code] * demand_change/1000
        
        # Remove zero or near-zero impacts and NaN values
        #significant_impacts = job_impacts[(abs(job_impacts) > 1e-6) & pd.notna(job_impacts)]
        significant_impacts = job_impacts
        # Create results with sector names (using sub-sector mapping for job results)
        results = []
        for sector_code, impact in significant_impacts.items():
            # For job coefficients, sector_code represents the sub-sector experiencing job impact
            # Use subsectormap to get proper sub-sector names
            if sector_code in self.subsector_to_name:
                sector_name = self.subsector_to_name.get(sector_code, f"Sub-sector {sector_code}")
            else:
                sector_name = f"Sub-sector {sector_code}"  # Fallback if name not found
            
            results.append({
                'sector_code': sector_code,
                'sector_name': sector_name,
                'impact': impact
            })
        
        # Sort by absolute impact (descending)
        results.sort(key=lambda x: abs(x['impact']), reverse=True)
        
        # Calculate summary statistics
        total_impact = sum([r['impact'] for r in results])
        
        return {
            'target_sector': target_sector,
            'target_product': target_product,
            'subsector_code': subsector_code,
            'demand_change': demand_change,
            'coeff_type': coeff_type,
            'coeff_name': coeff_name,
            'impacts': results,
            'total_impact': total_impact,
            'num_affected_sectors': len(results)
        }

    def aggregate_to_code_h(self, results: Dict) -> Dict:
        """
        Aggregate basic sector results to code_h level.

        Args:
            results: Dictionary from calculate_direct_effects

        Returns:
            Dictionary with aggregated results by code_h
        """
        code_h_impacts = {}

        for impact in results['impacts']:
            sector_code = impact['sector_code']
            impact_value = impact['impact']

            # Get code_h for this basic sector
            if sector_code in self.basic_to_code_h:
                code_h = self.basic_to_code_h[sector_code]
                product_h = self.code_h_to_product_h.get(code_h, f"Category {code_h}")

                if code_h not in code_h_impacts:
                    code_h_impacts[code_h] = {
                        'code_h': code_h,
                        'product_h': product_h,
                        'impact': 0,
                        'sector_count': 0
                    }

                code_h_impacts[code_h]['impact'] += impact_value
                code_h_impacts[code_h]['sector_count'] += 1

        # Convert to list and sort
        aggregated_results = list(code_h_impacts.values())
        aggregated_results.sort(key=lambda x: abs(x['impact']), reverse=True)

        total_impact = sum([r['impact'] for r in aggregated_results])

        return {
            'target_sector': results['target_sector'],
            'target_product': results['target_product'],
            'demand_change': results['demand_change'],
            'coeff_type': results['coeff_type'],
            'coeff_name': results['coeff_name'],
            'aggregation_level': 'code_h',
            'impacts': aggregated_results,
            'total_impact': total_impact,
            'num_affected_categories': len(aggregated_results)
        }

    def calculate_effects_by_code_h(self, target_sector, demand_change: float, coeff_type: str = 'indirect_prod', quiet: bool = False) -> Dict:
        """
        Calculate effects and automatically aggregate to code_h level.

        Args:
            target_sector: Sector code (string like "0111" or integer like 2711)
            demand_change: Change in final demand (positive or negative)
            coeff_type: Type of coefficients to use
            quiet: If True, suppress print output

        Returns:
            Dictionary with code_h aggregated results
        """
        # First calculate at basic sector level
        basic_results = self.calculate_direct_effects(target_sector, demand_change, coeff_type, quiet=quiet)

        # Then aggregate to code_h level
        return self.aggregate_to_code_h(basic_results)

    def display_code_h_results(self, results: Dict):
        """Display code_h aggregated analysis results."""
        print(f"\n{'='*70}")
        print(f"CODE_H AGGREGATED EFFECTS - {results['coeff_name'].upper()}")
        print(f"{'='*70}")
        print(f"Target Sector: {results['target_sector']} - {results['target_product']}")
        print(f"Demand Change: {results['demand_change']:,.0f}")
        print(f"Coefficient Type: {results['coeff_type']} ({results['coeff_name']})")
        print(f"Total Impact: {results['total_impact']:,.2f}")
        print(f"Affected Categories: {results['num_affected_categories']}")

        print(f"\n{'All Categories:':<70}")
        print(f"{'Code_H':<8} {'Category':<30} {'Sectors':>10} {'Impact':>18}")
        print("-" * 70)

        for impact in results['impacts']:
            print(f"{impact['code_h']:<8} {impact['product_h']:<30} {impact['sector_count']:>10} {impact['impact']:>18,.2f}")

    def calculate_all_effects(self, selected_sector, demand_change: float, quiet: bool = True) -> tuple:
        """
        Calculate all coefficient effects for a given sector and demand change.

        Args:
            selected_sector: Sector code (string like "0111" or integer like 2711)
            demand_change: Change in final demand (positive or negative)
            quiet: If True, suppress print output (default True for GUI use)

        Returns:
            Tuple of (all_results, coefficient_types, coeff_names)
            - all_results: Dictionary with results for each coefficient type
            - coefficient_types: List of coefficient types analyzed
            - coeff_names: Dictionary mapping coefficient types to display names
        """
        # Define coefficient types and their display names
        coefficient_types = ["indirect_prod", "indirect_import", "value_added", "jobcoeff", "directemploycoeff"]
        coeff_names = {
            "indirect_prod": "Production-inducing",
            "indirect_import": "Import-inducing",
            "value_added": "Value-Added",
            "jobcoeff": "Total Job Creation",
            "directemploycoeff": "Direct Employment"
        }

        # Calculate all coefficient effects
        all_results = {}
        for coeff_type in coefficient_types:
            try:
                results = self.calculate_direct_effects(
                    selected_sector,
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
        Create combined data from all analysis results with code_h and product_h mappings.

        Args:
            all_results: Dictionary of results from calculate_direct_effects for each coefficient type
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
                    # Get code_h and product_h if available
                    sector_code = impact['sector_code']
                    code_h = self.basic_to_code_h.get(sector_code, '') if hasattr(self, 'basic_to_code_h') else ''
                    product_h = self.code_h_to_product_h.get(code_h, '') if code_h and hasattr(self, 'code_h_to_product_h') else ''

                    combined_data.append({
                        'coefficient_type': coeff_type,
                        'coefficient_name': coeff_names[coeff_type],
                        'sector_code': sector_code,
                        'sector_name': impact['sector_name'],
                        'code_h': code_h,
                        'product_h': product_h,
                        'impact': impact['impact']
                    })

        return combined_data

if __name__ == "__main__":
    analyzer = IOTableAnalyzer()

    # Test calculate_all_effects
    print("\n" + "="*70)
    print("TEST: calculate_all_effects()")
    print("="*70)
    target_sector = "1610"
    demand_change = 345000

    all_results, coefficient_types, coeff_names = analyzer.calculate_all_effects(
        target_sector,
        demand_change,
        quiet=False
    )
    for coeff_type in coefficient_types:
        print(f"\n--- Results for coefficient type: {coeff_type} ({coeff_names[coeff_type]}) ---")
        results = all_results[coeff_type]
        if results:
            analyzer.display_results(results)
        else:
            print("No results available.")

    # Test create_combined_data
    print("\n" + "="*70)
    print("TEST: create_combined_data()")
    print("="*70)
    combined_data = analyzer.create_combined_data(all_results, coefficient_types, coeff_names)
    # Print header
    if combined_data:
        print(f"{'CoeffType':<18} {'CoeffName':<18} {'SectorCode':<10} {'SectorName':<30} {'Code_H':<8} {'Product_H':<16} {'Impact':>12}")
        print("-" * 110)
        for row in combined_data[:10]:  # Only show up to 10 rows for brevity
            print(f"{row['coefficient_type']:<18} {row['coefficient_name']:<18} {row['sector_code']:<10} {row['sector_name']:<30} {row['code_h']:<8} {row['product_h']:<16} {row['impact']:>12,.2f}")
        if len(combined_data) > 10:
            print(f"... ({len(combined_data)} total rows)")
    else:
        print("No combined data available.")