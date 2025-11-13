import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from libs.hydrogen_analyzer import HydrogenTableAnalyzer
from libs.io_analyzer import IOTableAnalyzer

class ScenarioAnalyzer:
    def __init__(self, scenarios_file: str = 'data/Data_v10.xlsx'):
        """Initialize the Scenario Analyzer with scenarios data."""
        self.scenarios_file = scenarios_file
        self.scenarios_data = None
        self.hydrogen_analyzer = None
        self.io_analyzer = None
        self.results = {}
        self.aggregated_results = {}
        self.load_scenarios()

    def load_scenarios(self):
        """Load scenarios from Excel file, reading all sheets that start with 'scenario'."""
        print("Loading scenarios data...")

        # Get all sheet names from the Excel file
        excel_file = pd.ExcelFile(self.scenarios_file)
        scenario_sheets = [sheet for sheet in excel_file.sheet_names if sheet.lower().startswith('scenario')]

        print(f"Found {len(scenario_sheets)} scenario sheets: {scenario_sheets}")

        # Store sheet names for later reference
        self.scenario_sheet_names = scenario_sheets

        # Load and combine all scenario sheets
        all_scenarios = []
        for sheet_name in scenario_sheets:
            df = pd.read_excel(self.scenarios_file, sheet_name=sheet_name)
            # Add a column to track which sheet this row came from
            df['scenario_sheet'] = sheet_name
            all_scenarios.append(df)
            print(f"  Loaded sheet '{sheet_name}' with {len(df)} rows")

        # Concatenate all scenario dataframes
        self.scenarios_data = pd.concat(all_scenarios, ignore_index=True)

        # Convert mixed data types to strings for consistency
        self.scenarios_data['input'] = self.scenarios_data['input'].astype(str)
        self.scenarios_data['sector'] = self.scenarios_data['sector'].astype(str)

        print(f"Total loaded: {len(self.scenarios_data)} scenarios")
        print(f"Years covered: {self.scenarios_data.columns[2:].tolist()}")

    def initialize_analyzers(self):
        """Initialize the hydrogen and IO table analyzers."""
        print("Initializing analyzers...")
        self.hydrogen_analyzer = HydrogenTableAnalyzer()
        self.io_analyzer = IOTableAnalyzer()

    def run_all_scenarios(self, effect_types: List[str] = None):
        """
        Run analysis for all scenarios across all years.

        Args:
            effect_types: List of effect types to analyze. If None, analyzes all available types.
        """
        if self.hydrogen_analyzer is None or self.io_analyzer is None:
            self.initialize_analyzers()
        if effect_types is None:
            # Default effect types based on available coefficient matrices
            effect_types = [
                'productioncoeff',  # (for hydrogen)
                'indirect_prod',  # (for IO)
                'indirect_import',  # (for IO)
                'jobcoeff',  # Job creation
                'valueaddedcoeff',  # Value added (hydrogen)
                'value_added',  # Value added (IO)
                'directemploycoeff' # Direct employment
            ]

        print(f"Running scenario analysis for effect types: {effect_types}")

        # Get year columns (exclude 'input' and 'sector' columns)
        year_columns = [col for col in self.scenarios_data.columns if isinstance(col, int)]

        # Initialize results structure
        for effect_type in effect_types:
            self.results[effect_type] = {}
            self.aggregated_results[effect_type] = {}

        # Process each scenario row
        for idx, scenario_row in self.scenarios_data.iterrows():
            input_table = scenario_row['input']
            sector = scenario_row['sector']

            print(idx)

            print(f"\nProcessing scenario {idx + 1}: {input_table} - {sector}")

            # Determine analyzer type based on input table
            is_hydrogen = 'hydrogen' in input_table.lower()

            # Process each year for this scenario
            for year in year_columns:
                demand_change = scenario_row[year]

                if pd.isna(demand_change) or demand_change == 0:
                    continue

                print(f"  Processing year {year}: demand change = {demand_change}")

                # Run analysis for each effect type
                for effect_type in effect_types:
                    try:
                        if is_hydrogen:
                            # Use hydrogen analyzer
                            if effect_type in ['productioncoeff', 'valueaddedcoeff', 'jobcoeff', 'directemploycoeff']:
                                result = self.hydrogen_analyzer.calculate_hydrogen_effects(
                                    scenario=sector,
                                    demand_change=demand_change,
                                    coeff_type=effect_type,
                                    quiet=True
                                )
                                self._store_result(effect_type, year, idx, result, is_hydrogen=True)

                        else:
                            # Use IO analyzer - convert sector to proper format for IO table
                            # try:
                            #     # Try to convert sector to int if it's a numeric string
                            #     if sector.isdigit():
                            #         target_sector = int(sector)
                            #     else:
                            #         target_sector = sector
                            # except (ValueError, AttributeError):
                            #     target_sector = sector

                            if effect_type in [ 'indirect_prod', 'indirect_import', 'value_added', 'jobcoeff', 'directemploycoeff']:
                                result = self.io_analyzer.calculate_direct_effects(
                                    target_sector=sector,
                                    demand_change=demand_change,
                                    coeff_type=effect_type,
                                    quiet=True
                                )
                                self._store_result(effect_type, year, idx, result, is_hydrogen=False)

                    except Exception as e:
                        print(f"    Error in {effect_type}: {str(e)}")
                        continue

        # Aggregate results by year and effect type
        self._aggregate_results(year_columns, effect_types)
        print("\nScenario analysis complete!")

    def _store_result(self, effect_type: str, year: int, scenario_idx: int, result: Dict, is_hydrogen: bool):
        """Store individual scenario result."""
        if year not in self.results[effect_type]:
            self.results[effect_type][year] = {}

        
        total_impact = 0 # 기본값 설정
        # if 'total_job_impact' in result and result['total_job_impact'] != 0:
        #     total_impact = result['total_job_impact']
        # elif 'total_economic_impact' in result:
        #     total_impact = result['total_economic_impact']

        scenario_key = f"scenario_{scenario_idx}"
        self.results[effect_type][year][scenario_key] = {
            'result': result,
            'is_hydrogen': is_hydrogen,
            'total_impact': result['total_impact'],
            'num_affected_sectors': result['num_affected_sectors']
        }


    def _aggregate_results(self, year_columns: List[int], effect_types: List[str]):
        """Aggregate results across all scenarios for each year and effect type."""
        print("Aggregating results...")

        for effect_type in effect_types:
            self.aggregated_results[effect_type] = {}

            for year in year_columns:
                if year not in self.results[effect_type]:
                    continue

                # Aggregate all scenarios for this year and effect type
                year_results = self.results[effect_type][year]

                # Combine all sector impacts
                all_sector_impacts = {}
                total_aggregate_impact = 0
                scenario_count = 0

                for scenario_key, scenario_data in year_results.items():
                    result = scenario_data['result']
                    total_aggregate_impact += result['total_impact']
                    scenario_count += 1

                    # Aggregate sector-level impacts
                    for impact in result['impacts']:
                        sector_code = str(impact['sector_code'])  # Ensure string
                        sector_name = impact['sector_name']
                        impact_value = impact['impact']

                        if sector_code not in all_sector_impacts:
                            all_sector_impacts[sector_code] = {
                                'sector_name': sector_name,
                                'total_impact': 0,
                                'scenario_count': 0
                            }

                        all_sector_impacts[sector_code]['total_impact'] += impact_value
                        all_sector_impacts[sector_code]['scenario_count'] += 1

                # Convert to sorted list
                aggregated_impacts = []
                for sector_code, data in all_sector_impacts.items():
                    aggregated_impacts.append({
                        'sector_code': sector_code,
                        'sector_name': data['sector_name'],
                        'total_impact': data['total_impact'],
                        'avg_impact': data['total_impact'] / data['scenario_count'],
                        'scenario_count': data['scenario_count']
                    })

                # Sort by total impact
                aggregated_impacts.sort(key=lambda x: abs(x['total_impact']), reverse=True)

                self.aggregated_results[effect_type][year] = {
                    'total_aggregate_impact': total_aggregate_impact,
                    'scenario_count': scenario_count,
                    'avg_aggregate_impact': total_aggregate_impact / scenario_count if scenario_count > 0 else 0,
                    'sector_impacts': aggregated_impacts,
                    'num_affected_sectors': len(aggregated_impacts)
                }

    def create_summary_tables(self, output_dir: str = 'output'):
        """Create Excel files with summary tables for each effect type."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        print(f"Creating summary tables in {output_dir}/...")

        for effect_type in self.aggregated_results.keys():
            if not self.aggregated_results[effect_type]:
                continue

            # Create Excel file for this effect type
            filename = f"{output_dir}/scenario_analysis_{effect_type}.xlsx"

            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Summary by year (aggregated totals)
                summary_data = []
                for year, data in self.aggregated_results[effect_type].items():
                    summary_data.append({
                        'Year': year,
                        'Total_Impact': data['total_aggregate_impact'],
                        'Avg_Impact': data['avg_aggregate_impact'],
                        'Scenario_Count': data['scenario_count'],
                        'Affected_Sectors': data['num_affected_sectors']
                    })

                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary_by_Year', index=False)

                # Detailed sector impacts by year
                all_years = sorted(self.aggregated_results[effect_type].keys())
                all_sectors = set()

                # Collect all unique sectors
                for year_data in self.aggregated_results[effect_type].values():
                    for impact in year_data['sector_impacts']:
                        all_sectors.add(str(impact['sector_code']))

                # Create sector impact matrix
                sector_matrix = []
                for sector_code in sorted(all_sectors):
                    row = {'Sector_Code': sector_code, 'Sector_Name': ''}

                    # Add code_h and product_h columns if IO analyzer is available and this is not a job coefficient
                    if self.io_analyzer and effect_type not in ['jobcoeff', 'directemploycoeff']:
                        code_h = self.io_analyzer.basic_to_code_h.get(sector_code, '')
                        product_h = self.io_analyzer.code_h_to_product_h.get(code_h, '') if code_h else ''
                        row['Code_H'] = code_h
                        row['Category_H'] = product_h

                    for year in all_years:
                        year_data = self.aggregated_results[effect_type][year]
                        impact_value = 0
                        sector_name = ''

                        for impact in year_data['sector_impacts']:
                            if str(impact['sector_code']) == sector_code:
                                impact_value = impact['total_impact']
                                sector_name = impact['sector_name']
                                break

                        row[f'Year_{year}'] = impact_value
                        if not row['Sector_Name']:
                            row['Sector_Name'] = sector_name

                    sector_matrix.append(row)

                sector_df = pd.DataFrame(sector_matrix)

                # Reorder columns to have Code_H and Category_H after Sector_Name
                if self.io_analyzer and effect_type not in ['jobcoeff', 'directemploycoeff']:
                    cols = ['Sector_Code', 'Sector_Name', 'Code_H', 'Category_H'] + [col for col in sector_df.columns if col.startswith('Year_')]
                    sector_df = sector_df[cols]

                sector_df.to_excel(writer, sheet_name='Sector_Impacts_by_Year', index=False)

            print(f"Created {filename}")

    def save_individual_scenario_csvs(self, output_dir: str = 'output'):
        """
        Save individual CSV files for each scenario (input table + sector combination).
        Files are named: scenario_{effect_type}_{sector}_{input_table}.csv
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print(f"Saving individual scenario CSV files to {output_dir}/...")

        # Group results by (effect_type, input_table, sector)
        scenario_data_map = {}

        for effect_type in self.results.keys():
            if not self.results[effect_type]:
                continue

            for year in self.results[effect_type].keys():
                for scenario_key, scenario_result in self.results[effect_type][year].items():
                    # Extract scenario index from scenario_key (e.g., "scenario_0" -> 0)
                    scenario_idx = int(scenario_key.split('_')[1])

                    # Get scenario metadata from scenarios_data
                    scenario_row = self.scenarios_data.iloc[scenario_idx]
                    input_table = scenario_row['input']
                    sector = str(scenario_row['sector'])

                    # Create unique key for this scenario
                    unique_key = (effect_type, sector, input_table)

                    if unique_key not in scenario_data_map:
                        scenario_data_map[unique_key] = {}

                    # Store results for this year
                    scenario_data_map[unique_key][year] = scenario_result['result']

        # Now create CSV files for each unique scenario
        for (effect_type, sector, input_table), year_results in scenario_data_map.items():
            # Sort years
            sorted_years = sorted(year_results.keys())

            # Collect all unique output sectors across all years
            all_output_sectors = {}
            for year, result in year_results.items():
                for impact in result['impacts']:
                    sector_code = str(impact['sector_code'])
                    sector_name = impact['sector_name']
                    if sector_code not in all_output_sectors:
                        all_output_sectors[sector_code] = sector_name

            # Create matrix data (rows = output sectors, columns = years)
            matrix_data = []
            for sector_code, sector_name in all_output_sectors.items():
                row = {
                    'Sector_Code': sector_code,
                    'Sector_Name': sector_name
                }

                # Add impact values for each year
                for year in sorted_years:
                    impact_value = 0.0
                    result = year_results[year]

                    for impact in result['impacts']:
                        if str(impact['sector_code']) == sector_code:
                            impact_value = impact['impact']
                            break

                    row[str(year)] = impact_value

                matrix_data.append(row)

            # Create DataFrame
            if matrix_data:
                df = pd.DataFrame(matrix_data)

                # Sort by the latest year's absolute impact
                latest_year_col = str(sorted_years[-1])
                if latest_year_col in df.columns:
                    df['abs_latest'] = df[latest_year_col].abs()
                    df = df.sort_values('abs_latest', ascending=False)
                    df = df.drop('abs_latest', axis=1)

                # Clean up the input_table name for filename
                input_table_clean = input_table.replace('.xlsx', '').replace(' ', '_')

                # Create filename: scenario_{effect_type}_{sector}_{input_table}.csv
                filename = f"{output_dir}/scenario_{effect_type}_{sector}_{input_table_clean}.csv"

                # Save to CSV with UTF-8-BOM encoding for proper Korean character display in Excel
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"Created {filename}")

        print(f"Saved {len(scenario_data_map)} individual scenario CSV files.")

    
    def integrate_sectors_1610_4506(self, effect_types: List[str] = None):
        """
        Integrate results from sectors 1610 and 4506 (IO Table) by summing impacts for each sector code.

        Args:
            effect_types: List of effect types to integrate. If None, uses IO table effect types.

        Returns:
            Dictionary with integrated results for each effect type and year
        """
        if effect_types is None:
            effect_types = ['indirect_prod', 'indirect_import', 'value_added', 'jobcoeff', 'directemploycoeff']

        print("Integrating sectors 1610 and 4506 (sector-by-sector)...")

        # Find scenario indices for 1610 and 4506
        idx_1610 = None
        idx_4506 = None

        for idx, row in self.scenarios_data.iterrows():
            sector = str(row['sector'])
            if sector == '1610':
                idx_1610 = idx
            elif sector == '4506':
                idx_4506 = idx

        if idx_1610 is None or idx_4506 is None:
            print("Could not find scenarios for sectors 1610 and/or 4506")
            return None

        # Create integrated results
        integrated_results = {}

        for effect_type in effect_types:
            if effect_type not in self.results:
                continue

            integrated_results[effect_type] = {}

            for year in self.results[effect_type].keys():
                scenario_key_1610 = f"scenario_{idx_1610}"
                scenario_key_4506 = f"scenario_{idx_4506}"

                # Get results for both sectors
                result_1610 = self.results[effect_type][year].get(scenario_key_1610)
                result_4506 = self.results[effect_type][year].get(scenario_key_4506)

                if not result_1610 or not result_4506:
                    continue

                # Merge the impacts sector-by-sector (summing for each sector code)
                combined_impacts = {}

                # Add impacts from 1610
                for impact in result_1610['result']['impacts']:
                    sector_code = str(impact['sector_code'])
                    combined_impacts[sector_code] = {
                        'sector_code': sector_code,
                        'sector_name': impact['sector_name'],
                        'impact': impact['impact']
                    }

                # Add impacts from 4506 (sum if sector exists, add new if not)
                for impact in result_4506['result']['impacts']:
                    sector_code = str(impact['sector_code'])
                    if sector_code in combined_impacts:
                        # Sum the impacts for this sector
                        combined_impacts[sector_code]['impact'] += impact['impact']
                    else:
                        # New sector from 4506
                        combined_impacts[sector_code] = {
                            'sector_code': sector_code,
                            'sector_name': impact['sector_name'],
                            'impact': impact['impact']
                        }

                # Convert to list and sort by absolute impact
                impacts_list = list(combined_impacts.values())
                impacts_list.sort(key=lambda x: abs(x['impact']), reverse=True)

                # Calculate total impact
                total_impact = sum([imp['impact'] for imp in impacts_list])

                integrated_results[effect_type][year] = {
                    'impacts': impacts_list,
                    'total_impact': total_impact,
                    'num_affected_sectors': len(impacts_list)
                }

        return integrated_results

    def integrate_hydrogen_H2S_H2T(self, effect_types: List[str] = None):
        """
        Integrate results from hydrogen scenarios H2S and H2T by summing impacts for each sector code.

        Args:
            effect_types: List of effect types to integrate. If None, uses hydrogen effect types.

        Returns:
            Dictionary with integrated results for each effect type and year
        """
        if effect_types is None:
            effect_types = ['productioncoeff', 'valueaddedcoeff', 'jobcoeff', 'directemploycoeff']

        print("Integrating hydrogen scenarios H2S and H2T (sector-by-sector)...")

        # Find scenario indices for H2S and H2T
        idx_H2S = None
        idx_H2T = None

        for idx, row in self.scenarios_data.iterrows():
            sector = str(row['sector'])
            input_table = str(row['input'])
            if 'hydrogen' in input_table.lower():
                if sector == 'H2S':
                    idx_H2S = idx
                elif sector == 'H2T':
                    idx_H2T = idx

        if idx_H2S is None or idx_H2T is None:
            print("Could not find scenarios for H2S and/or H2T")
            return None

        # Create integrated results
        integrated_results = {}

        for effect_type in effect_types:
            if effect_type not in self.results:
                continue

            integrated_results[effect_type] = {}

            for year in self.results[effect_type].keys():
                scenario_key_H2S = f"scenario_{idx_H2S}"
                scenario_key_H2T = f"scenario_{idx_H2T}"

                # Get results for both scenarios
                result_H2S = self.results[effect_type][year].get(scenario_key_H2S)
                result_H2T = self.results[effect_type][year].get(scenario_key_H2T)

                if not result_H2S or not result_H2T:
                    continue

                # Merge the impacts sector-by-sector (summing for each sector code)
                combined_impacts = {}

                # Add impacts from H2S
                for impact in result_H2S['result']['impacts']:
                    sector_code = str(impact['sector_code'])
                    combined_impacts[sector_code] = {
                        'sector_code': sector_code,
                        'sector_name': impact['sector_name'],
                        'impact': impact['impact']
                    }

                # Add impacts from H2T (sum if sector exists, add new if not)
                for impact in result_H2T['result']['impacts']:
                    sector_code = str(impact['sector_code'])
                    if sector_code in combined_impacts:
                        # Sum the impacts for this sector
                        combined_impacts[sector_code]['impact'] += impact['impact']
                    else:
                        # New sector from H2T
                        combined_impacts[sector_code] = {
                            'sector_code': sector_code,
                            'sector_name': impact['sector_name'],
                            'impact': impact['impact']
                        }

                # Convert to list and sort by absolute impact
                impacts_list = list(combined_impacts.values())
                impacts_list.sort(key=lambda x: abs(x['impact']), reverse=True)

                # Calculate total impact
                total_impact = sum([imp['impact'] for imp in impacts_list])

                integrated_results[effect_type][year] = {
                    'impacts': impacts_list,
                    'total_impact': total_impact,
                    'num_affected_sectors': len(impacts_list)
                }

        return integrated_results

    def save_integrated_scenarios(self, output_dir: str = 'output'):
        """
        Save integrated scenario results to CSV files for both 1610+4506 and H2S+H2T.
        Each CSV contains sector-by-sector impacts across all years.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print(f"Saving integrated scenario results to {output_dir}/...")

        # Integrate 1610 + 4506
        integrated_io = self.integrate_sectors_1610_4506()
        if integrated_io:
            self._save_integrated_csv(integrated_io, '1610+4506', 'iotable_2023', output_dir)

        # Integrate H2S + H2T
        integrated_h2 = self.integrate_hydrogen_H2S_H2T()
        if integrated_h2:
            self._save_integrated_csv(integrated_h2, 'H2S+H2T', 'hydrogentable_2023', output_dir)

        print("Integrated scenarios saved successfully!")

    def _save_integrated_csv(self, integrated_results: Dict, scenario_name: str,
                            input_table: str, output_dir: str):
        """
        Helper function to save integrated results to CSV files.

        Args:
            integrated_results: Dictionary of integrated results (from integrate methods)
            scenario_name: Name of the integrated scenario (e.g., '1610_4506' or 'H2S_H2T')
            input_table: Input table name
            output_dir: Output directory
        """
        for effect_type, year_results in integrated_results.items():
            if not year_results:
                continue

            # Sort years
            sorted_years = sorted(year_results.keys())

            # Collect all unique output sectors across all years
            all_output_sectors = {}
            for year, result in year_results.items():
                for impact in result['impacts']:
                    sector_code = str(impact['sector_code'])
                    sector_name = impact['sector_name']
                    if sector_code not in all_output_sectors:
                        all_output_sectors[sector_code] = sector_name

            # Create matrix data (rows = output sectors, columns = years)
            matrix_data = []
            for sector_code, sector_name in all_output_sectors.items():
                row = {
                    'Sector_Code': sector_code,
                    'Sector_Name': sector_name
                }

                # Add impact values for each year
                for year in sorted_years:
                    impact_value = 0.0
                    result = year_results[year]

                    for impact in result['impacts']:
                        if str(impact['sector_code']) == sector_code:
                            impact_value = impact['impact']
                            break

                    row[str(year)] = impact_value

                matrix_data.append(row)

            # Create DataFrame
            if matrix_data:
                df = pd.DataFrame(matrix_data)

                # Sort by the latest year's absolute impact
                latest_year_col = str(sorted_years[-1])
                if latest_year_col in df.columns:
                    df['abs_latest'] = df[latest_year_col].abs()
                    df = df.sort_values('abs_latest', ascending=False)
                    df = df.drop('abs_latest', axis=1)

                # Create filename
                filename = f"{output_dir}/scenario_{effect_type}_{scenario_name}_{input_table}.csv"

                # Save to CSV with UTF-8-BOM encoding
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"Created {filename}")

    def aggregate_integrated_by_code_h(self, integrated_results: Dict) -> Dict:
        """
        Aggregate integrated results by code_h category.

        Args:
            integrated_results: Dictionary from integrate_sectors_1610_4506() or integrate_hydrogen_H2S_H2T()

        Returns:
            Dictionary with code_h aggregated results for each effect type and year
        """
        if self.io_analyzer is None:
            self.initialize_analyzers()

        aggregated_results = {}

        for effect_type, year_results in integrated_results.items():
            if not year_results:
                continue

            aggregated_results[effect_type] = {}

            for year, result in year_results.items():
                # Aggregate by code_h
                code_h_impacts = {}

                for impact in result['impacts']:
                    sector_code = str(impact['sector_code'])
                    impact_value = impact['impact']

                    # Get code_h for this sector
                    if sector_code in self.io_analyzer.basic_to_code_h:
                        code_h = self.io_analyzer.basic_to_code_h[sector_code]
                        product_h = self.io_analyzer.code_h_to_product_h.get(code_h, f"Category {code_h}")

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
                impacts_list = list(code_h_impacts.values())
                impacts_list.sort(key=lambda x: abs(x['impact']), reverse=True)

                # Calculate total impact
                total_impact = sum([imp['impact'] for imp in impacts_list])

                aggregated_results[effect_type][year] = {
                    'impacts': impacts_list,
                    'total_impact': total_impact,
                    'num_affected_categories': len(impacts_list)
                }

        return aggregated_results

    def save_integrated_code_h_results(self, output_dir: str = 'output'):
        """
        Save code_h aggregated integrated scenario results for both 1610+4506 and H2S+H2T.
        Creates CSV files with code_h categories as rows and years as columns.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print(f"Saving code_h aggregated integrated results to {output_dir}/...")

        # Process 1610 + 4506
        print("\nProcessing 1610 + 4506 integration with code_h aggregation...")
        integrated_io = self.integrate_sectors_1610_4506()
        if integrated_io:
            aggregated_io = self.aggregate_integrated_by_code_h(integrated_io)
            self._save_code_h_csv(aggregated_io, '1610_4506', 'iotable_2023', output_dir)

        # Process H2S + H2T
        print("\nProcessing H2S + H2T integration with code_h aggregation...")
        integrated_h2 = self.integrate_hydrogen_H2S_H2T()
        if integrated_h2:
            aggregated_h2 = self.aggregate_integrated_by_code_h(integrated_h2)
            self._save_code_h_csv(aggregated_h2, 'H2S_H2T', 'hydrogentable_2023', output_dir)

        print("\nCode_h aggregated integrated scenarios saved successfully!")

    def _save_code_h_csv(self, aggregated_results: Dict, scenario_name: str,
                         input_table: str, output_dir: str):
        """
        Helper function to save code_h aggregated results to CSV files.

        Args:
            aggregated_results: Dictionary from aggregate_integrated_by_code_h()
            scenario_name: Name of the integrated scenario (e.g., '1610_4506' or 'H2S_H2T')
            input_table: Input table name
            output_dir: Output directory
        """
        for effect_type, year_results in aggregated_results.items():
            if not year_results:
                continue

            # Sort years
            sorted_years = sorted(year_results.keys())

            # Collect all unique code_h categories across all years
            all_code_h = {}
            for year, result in year_results.items():
                for impact in result['impacts']:
                    code_h = impact['code_h']
                    product_h = impact['product_h']
                    if code_h not in all_code_h:
                        all_code_h[code_h] = product_h

            # Create matrix data (rows = code_h categories, columns = years)
            matrix_data = []
            for code_h, product_h in sorted(all_code_h.items()):
                row = {
                    'Code_H': code_h,
                    'Category_H': product_h
                }

                # Add impact values for each year
                for year in sorted_years:
                    impact_value = 0.0
                    result = year_results[year]

                    for impact in result['impacts']:
                        if impact['code_h'] == code_h:
                            impact_value = impact['impact']
                            break

                    row[str(year)] = impact_value

                matrix_data.append(row)

            # Create DataFrame
            if matrix_data:
                df = pd.DataFrame(matrix_data)

                # Sort by the latest year's absolute impact
                latest_year_col = str(sorted_years[-1])
                if latest_year_col in df.columns:
                    df['abs_latest'] = df[latest_year_col].abs()
                    df = df.sort_values('abs_latest', ascending=False)
                    df = df.drop('abs_latest', axis=1)

                # Create filename with code_h suffix
                filename = f"{output_dir}/scenario_{effect_type}_{scenario_name}_{input_table}_code_h.csv"

                # Save to CSV with UTF-8-BOM encoding
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"Created {filename}")

    def create_comprehensive_summary_tables(self, years: List[int] = None, output_dir: str = 'output'):
        """
        Create comprehensive summary tables for all scenarios (individual + integrated).
        Creates 6 summary tables showing total impacts across all years:
        1. Sector 1610 (IO Table)
        2. Sector 4506 (IO Table)
        3. H2S (Hydrogen)
        4. H2T (Hydrogen)
        5. Integrated 1610 & 4506
        6. Integrated H2S & H2T

        Args:
            years: List of years to include (default: all available years from results)
            output_dir: Directory to save the Excel file
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # If no years specified, get all years from results
        if years is None:
            years_set = set()
            for effect_type in self.results.keys():
                years_set.update(self.results[effect_type].keys())
            years = sorted(years_set)

            if not years:
                print("No results found. Please run run_all_scenarios() first.")
                return {}

            print(f"Using all available years: {years}")

        print("Creating comprehensive summary tables...")
        print("=" * 80)

        summary_tables = {}

        # Helper function to create summary from results dictionary
        def create_summary_from_results(results_dict, effect_map, is_hydrogen=False):
            """
            Create summary table from a results dictionary.

            Args:
                results_dict: Dictionary with structure {effect_type: {year: {'total_impact': value}}}
                effect_map: Mapping of effect types to use
                is_hydrogen: Whether this is hydrogen data (affects units)
            """
            summary_rows = []

            for year in years:
                row = {'Year': year}

                for effect_type, column_name in effect_map.items():
                    if effect_type in results_dict and year in results_dict[effect_type]:
                        total = results_dict[effect_type][year]['total_impact']

                        # Convert to appropriate units
                        if 'billion won' in column_name.lower():
                            row[column_name] = total / 1000
                        else:
                            row[column_name] = total
                    else:
                        row[column_name] = 0 if 'N/A' not in column_name else 'N/A'

                summary_rows.append(row)

            return pd.DataFrame(summary_rows).set_index('Year')

        # Find scenario indices
        scenario_indices = {}
        for idx, row in self.scenarios_data.iterrows():
            sector = str(row['sector'])
            if sector in ['1610', '4506', 'H2S', 'H2T']:
                scenario_indices[sector] = idx

        # Define effect type mappings
        io_effect_map = {
            'indirect_prod': 'Indirect Production (billion won)',
            'indirect_import': 'Import (billion won)',
            'value_added': 'Value Added (billion won)',
            'jobcoeff': 'Job Creation (person/billion won)',
            'directemploycoeff': 'Direct Employment (person/billion won)'
        }

        hydrogen_effect_map = {
            'productioncoeff': 'Indirect Production (billion won)',
            'valueaddedcoeff': 'Value Added (billion won)',
            'jobcoeff': 'Job Creation (billion won)',
            'directemploycoeff': 'Direct Employment (person/billion won)'
        }

        # Create individual scenario summaries by extracting from self.results
        for sector_name, sheet_name, effect_map, is_hydrogen in [
            ('1610', '1610_IO_Table', io_effect_map, False),
            ('4506', '4506_IO_Table', io_effect_map, False),
            ('H2S', 'H2S_Hydrogen', hydrogen_effect_map, True),
            ('H2T', 'H2T_Hydrogen', hydrogen_effect_map, True)
        ]:
            if sector_name not in scenario_indices:
                continue

            print(f"\n{len(summary_tables)+1}. Creating summary for {sector_name}...")

            idx = scenario_indices[sector_name]
            scenario_key = f"scenario_{idx}"

            # Extract this scenario's results from self.results
            scenario_results = {}
            for effect_type in effect_map.keys():
                if effect_type in self.results:
                    scenario_results[effect_type] = {}
                    for year in years:
                        if year in self.results[effect_type] and scenario_key in self.results[effect_type][year]:
                            scenario_results[effect_type][year] = self.results[effect_type][year][scenario_key]

            # Add 'N/A' for import in hydrogen scenarios
            if is_hydrogen:
                summary_df = create_summary_from_results(scenario_results, effect_map, is_hydrogen)
                summary_df['Import (billion won)'] = 'N/A'
                # Reorder columns
                cols = ['Indirect Production (billion won)', 'Import (billion won)', 'Value Added (billion won)',
                       'Job Creation (billion won)', 'Direct Employment (person/billion won)']
                summary_df = summary_df[[c for c in cols if c in summary_df.columns]]
            else:
                summary_df = create_summary_from_results(scenario_results, effect_map, is_hydrogen)

            summary_tables[sheet_name] = summary_df

        # Create integrated summaries using existing integration methods
        print(f"\n{len(summary_tables)+1}. Creating summary for Integrated 1610 & 4506...")
        integrated_io = self.integrate_sectors_1610_4506()
        if integrated_io:
            summary_tables['Integrated_1610_4506'] = create_summary_from_results(
                integrated_io, io_effect_map, is_hydrogen=False
            )

        print(f"\n{len(summary_tables)+1}. Creating summary for Integrated H2S & H2T...")
        integrated_h2 = self.integrate_hydrogen_H2S_H2T()
        if integrated_h2:
            summary_df = create_summary_from_results(integrated_h2, hydrogen_effect_map, is_hydrogen=True)
            summary_df['Import (billion won)'] = 'N/A'
            # Reorder columns
            cols = ['Indirect Production (billion won)', 'Import (billion won)', 'Value Added (billion won)',
                   'Job Creation (billion won)', 'Direct Employment (person/billion won)']
            summary_df = summary_df[[c for c in cols if c in summary_df.columns]]
            summary_tables['Integrated_H2S_H2T'] = summary_df

        # Export to Excel
        output_file = f'{output_dir}/summary_tables_{min(years)}_{max(years)}.xlsx'
        print(f"\nExporting all summary tables to {output_file}...")

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for sheet_name, df in summary_tables.items():
                df.to_excel(writer, sheet_name=sheet_name)
                print(f"  ✅ Created sheet: {sheet_name}")

        print("\n" + "=" * 80)
        print(f"Successfully exported all {len(summary_tables)} summary tables!")
        print(f"File location: {output_file}")
        print("=" * 80)

        return summary_tables

    def calculate_combined_job_creation(
        self,
        year: int,
        io_sectors: Optional[List[str]] = None,
        h2_sectors: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Calculate combined job creation for IO and hydrogen sectors.

        Args:
            year: Target year for the calculation.
            io_sectors: List of IO sector codes to include (default ['1610', '4506']).
            h2_sectors: List of hydrogen sector codes to include (default ['H2S', 'H2T']).

        Returns:
            Dictionary containing IO-only, H2-only, and combined totals.
        """
        io_sectors = io_sectors or ['1610', '4506']
        h2_sectors = h2_sectors or ['H2S', 'H2T']

        io_job_total = 0.0
        h2_job_total = 0.0

        # Sum IO job creation (jobcoeff) results in persons
        if 'jobcoeff' in self.results and year in self.results['jobcoeff']:
            for scenario_key, scenario_data in self.results['jobcoeff'][year].items():
                scenario_idx = int(scenario_key.split('_')[1])
                scenario_row = self.scenarios_data.iloc[scenario_idx]
                sector = str(scenario_row['sector'])
                is_hydrogen = scenario_data.get('is_hydrogen', False)

                if not is_hydrogen and sector in io_sectors:
                    io_job_total += scenario_data.get('total_impact', 0.0)

        # Sum hydrogen direct employment (persons)
        if 'directemploycoeff' in self.results and year in self.results['directemploycoeff']:
            for scenario_key, scenario_data in self.results['directemploycoeff'][year].items():
                scenario_idx = int(scenario_key.split('_')[1])
                scenario_row = self.scenarios_data.iloc[scenario_idx]
                sector = str(scenario_row['sector'])
                is_hydrogen = scenario_data.get('is_hydrogen', False)

                if (is_hydrogen or sector in h2_sectors) and sector in h2_sectors:
                    h2_job_total += scenario_data.get('total_impact', 0.0)

        combined_total = io_job_total + h2_job_total

        return {
            'io_total': io_job_total,
            'h2_total': h2_job_total,
            'combined_total': combined_total
        }


if __name__ == "__main__":
    # Example usage
    analyzer = ScenarioAnalyzer()

    # Run analysis for all scenarios
    analyzer.run_all_scenarios()

    # Save individual scenario results
    analyzer.save_individual_scenario_csvs(output_dir='output')

    # Save integrated scenarios (1610+4506 and H2S+H2T)
    analyzer.save_integrated_scenarios(output_dir='output')

    # Save code_h aggregated integrated scenarios
    analyzer.save_integrated_code_h_results(output_dir='output')


    print("\nScenario analysis complete! Check the 'output' directory for detailed results.")