from re import U
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import tempfile
from libs.io_analyzer import IOTableAnalyzer
from libs.hydrogen_analyzer import HydrogenTableAnalyzer
from libs.scenario_analyzer import ScenarioAnalyzer

# Define the target years used throughout the analysis
target_years = [2026, 2030, 2040, 2050]

# Configure Streamlit page
st.set_page_config(
    page_title="Hydrogen-reduced steel Input-Output Analysis",
    page_icon="🏭",
    layout="wide"
)

@st.cache_data
def load_analyzer():
    """Load the analyzer with caching to avoid reloading data."""
    return IOTableAnalyzer()

@st.cache_data
def load_hydrogen_analyzer():
    """Load the hydrogen analyzer with caching to avoid reloading data."""
    return HydrogenTableAnalyzer()

@st.cache_data
def load_scenario_analyzer():
    """Load the scenario analyzer with caching."""
    return ScenarioAnalyzer()

def show_scenarios():
    st.title("🎯 Scenarios")
    st.markdown("---")
    # Discover all Excel files in "data" folder whose name starts with "scenario_"
    data_folder = Path("data")
    scenario_files = sorted([f for f in data_folder.glob("scenario_*.xlsx")])
    if not scenario_files:
        st.warning("No scenario files found in the 'data' directory. Please add files named like scenario_x_xxxx.xlsx.")
        return

    # Load all scenario dataframes into a dictionary: {filename: dataframe}
    scenario_dataframes = {}
    for f in scenario_files:
        try:
            df = pd.read_excel(f)
            scenario_dataframes[f.name] = df
        except Exception as e:
            st.error(f"Failed to load {f.name}: {e}")

    # Optionally, display some feedback or all loaded dataframes
    st.info(f"Loaded {len(scenario_dataframes)} scenario files from the data folder.")
    # Preview option: Let user select and preview any scenario file
    if scenario_dataframes:
        file_to_preview = st.selectbox(
            "Select a scenario file to preview",
            list(scenario_dataframes.keys())
        )
        if st.checkbox("Preview selected scenario file"):
            st.dataframe(scenario_dataframes[file_to_preview])

    # Let user pick one or more scenarios
    scenario_names = [f.name for f in scenario_files]
    selected_scenario = st.selectbox("Select a Scenario File", scenario_names, help="Pick a scenario Excel file for batch analysis")

    selected_path = data_folder / selected_scenario

    st.info(f"You selected: `{selected_scenario}`")
    # User option: preview content
    if st.checkbox("Preview scenario file content"):
        try:
            scenario_df = pd.read_excel(selected_path)
            st.dataframe(scenario_df)
        except Exception as e:
            st.error(f"Failed to load scenario file: {e}")
            return


def run_scenario_analysis():
    """Run scenario analysis and store results in session state."""
    st.title("🚀 Run Scenario Analysis")
    st.markdown("---")
    st.markdown("Automatically analyze all scenarios from Data_v10.xlsx")

    # Fixed data file
    data_folder = Path("data")
    data_file = data_folder / "Data_v10.xlsx"

    # Check if file exists
    if not data_file.exists():
        st.error(f"❌ Data file not found: {data_file}")
        st.info("Please ensure Data_v10.xlsx exists in the data folder.")
        return

    # Show file info
    st.info(f"📄 Data file: `Data_v10.xlsx`")

    # Show which file is currently loaded
    if 'current_scenario_file' in st.session_state:
        current = st.session_state.current_scenario_file
        if current == "Data_v10.xlsx":
            st.success(f"✅ Currently loaded: `{current}`")
        else:
            st.warning(f"⚠️ Different file loaded: `{current}`")

    # Show preview option
    if st.checkbox("Preview scenario sheets", key="preview_run_analysis"):
        try:
            excel_file = pd.ExcelFile(data_file)
            scenario_sheets = [sheet for sheet in excel_file.sheet_names if sheet.lower().startswith('scenario')]
            st.write(f"**Found {len(scenario_sheets)} scenario sheets:**")
            for sheet_name in scenario_sheets:
                with st.expander(f"📋 {sheet_name}"):
                    scenario_df = pd.read_excel(data_file, sheet_name=sheet_name)
                    st.dataframe(scenario_df)
        except Exception as e:
            st.error(f"Failed to load data file: {e}")

    st.markdown("---")

    # Display current status
    if st.session_state.get('scenario_results'):
        st.info("✅ Scenario analysis results are available in session.")
        # Show which scenario sheets are loaded
        if 'scenario_analyzer' in st.session_state:
            analyzer = st.session_state.scenario_analyzer
            if hasattr(analyzer, 'scenario_sheet_names'):
                st.write(f"**Loaded scenario sheets:** {', '.join(analyzer.scenario_sheet_names)}")
    else:
        st.warning("⚠️ No scenario analysis results available yet.")

    # Run analysis button
    if st.button("🚀 Run Complete Scenario Analysis", type="primary", use_container_width=True):
        with st.spinner(f"Running scenario analysis for Data_v10.xlsx..."):
            try:
                # Initialize scenario analyzer with Data_v10.xlsx
                scenario_analyzer = ScenarioAnalyzer(scenarios_file=str(data_file))

                # Run all scenarios
                st.info("Running scenario analysis... This may take a few minutes.")
                scenario_analyzer.run_all_scenarios()

                # Store results in session state
                st.session_state.scenario_analyzer = scenario_analyzer
                st.session_state.scenario_results = scenario_analyzer.aggregated_results
                st.session_state.current_scenario_file = "Data_v10.xlsx"

                st.success(f"✅ Analysis complete for Data_v10.xlsx!")

                # Show summary of results
                st.markdown("### 📊 Analysis Summary")
                st.write(f"**Scenario sheets loaded:** {', '.join(scenario_analyzer.scenario_sheet_names)}")

                effect_types = list(scenario_analyzer.aggregated_results.keys())
                st.write(f"**Effect types analyzed:** {len(effect_types)}")
                st.write(f"**Effect types:** {', '.join(effect_types)}")

                # Count years
                years_set = set()
                for effect_type in effect_types:
                    years_set.update(scenario_analyzer.aggregated_results[effect_type].keys())
                st.write(f"**Years covered:** {sorted(years_set)}")

            except Exception as e:
                st.error(f"Error running analysis: {str(e)}")
                st.exception(e)


def filter_results_by_scenario_sheet(scenario_analyzer, sheet_name):
    """Filter and aggregate results for a specific scenario sheet."""
    filtered_results = {}

    # Get all effect types from the original results
    effect_types = list(scenario_analyzer.results.keys())

    for effect_type in effect_types:
        filtered_results[effect_type] = {}

        if effect_type not in scenario_analyzer.results:
            continue

        # Process each year
        for year, year_data in scenario_analyzer.results[effect_type].items():
            # Filter scenarios by sheet name
            filtered_year_data = {}

            for scenario_key, scenario_data in year_data.items():
                # Extract scenario index
                scenario_idx = int(scenario_key.split('_')[1])
                scenario_row = scenario_analyzer.scenarios_data.iloc[scenario_idx]

                # Check if this scenario matches our sheet filter
                if scenario_row['scenario_sheet'] == sheet_name:
                    filtered_year_data[scenario_key] = scenario_data

            if filtered_year_data:
                # Aggregate the filtered results for this year
                all_sector_impacts = {}
                total_aggregate_impact = 0
                scenario_count = 0

                for scenario_key, scenario_data in filtered_year_data.items():
                    result = scenario_data['result']
                    total_aggregate_impact += result['total_impact']
                    scenario_count += 1

                    # Aggregate sector-level impacts
                    for impact in result['impacts']:
                        sector_code = str(impact['sector_code'])
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

                aggregated_impacts.sort(key=lambda x: abs(x['total_impact']), reverse=True)

                filtered_results[effect_type][year] = {
                    'total_aggregate_impact': total_aggregate_impact,
                    'scenario_count': scenario_count,
                    'avg_aggregate_impact': total_aggregate_impact / scenario_count if scenario_count > 0 else 0,
                    'num_affected_sectors': len(all_sector_impacts),
                    'sector_impacts': aggregated_impacts
                }

    return filtered_results


def filter_results_by_sheet_and_sector(scenario_analyzer, sheet_name, sector_code):
    """Filter and aggregate results for a specific scenario sheet and sector."""
    filtered_results = {}

    # Get all effect types from the original results
    effect_types = list(scenario_analyzer.results.keys())

    for effect_type in effect_types:
        filtered_results[effect_type] = {}

        if effect_type not in scenario_analyzer.results:
            continue

        # Process each year
        for year, year_data in scenario_analyzer.results[effect_type].items():
            # Filter scenarios by sheet name AND sector
            filtered_year_data = {}

            for scenario_key, scenario_data in year_data.items():
                # Extract scenario index
                scenario_idx = int(scenario_key.split('_')[1])
                scenario_row = scenario_analyzer.scenarios_data.iloc[scenario_idx]

                # Check if this scenario matches our sheet and sector filter
                if (scenario_row['scenario_sheet'] == sheet_name and
                    str(scenario_row['sector']) == sector_code):
                    filtered_year_data[scenario_key] = scenario_data

            if filtered_year_data:
                # Aggregate the filtered results for this year
                all_sector_impacts = {}
                total_aggregate_impact = 0
                scenario_count = 0

                for scenario_key, scenario_data in filtered_year_data.items():
                    result = scenario_data['result']
                    total_aggregate_impact += result['total_impact']
                    scenario_count += 1

                    # Aggregate sector-level impacts
                    for impact in result['impacts']:
                        sector_code_impact = str(impact['sector_code'])
                        sector_name = impact['sector_name']
                        impact_value = impact['impact']

                        if sector_code_impact not in all_sector_impacts:
                            all_sector_impacts[sector_code_impact] = {
                                'sector_name': sector_name,
                                'total_impact': 0,
                                'scenario_count': 0
                            }

                        all_sector_impacts[sector_code_impact]['total_impact'] += impact_value
                        all_sector_impacts[sector_code_impact]['scenario_count'] += 1

                # Convert to sorted list
                aggregated_impacts = []
                for sector_code_impact, data in all_sector_impacts.items():
                    aggregated_impacts.append({
                        'sector_code': sector_code_impact,
                        'sector_name': data['sector_name'],
                        'total_impact': data['total_impact'],
                        'avg_impact': data['total_impact'] / data['scenario_count'],
                        'scenario_count': data['scenario_count']
                    })

                aggregated_impacts.sort(key=lambda x: abs(x['total_impact']), reverse=True)

                filtered_results[effect_type][year] = {
                    'total_aggregate_impact': total_aggregate_impact,
                    'scenario_count': scenario_count,
                    'avg_aggregate_impact': total_aggregate_impact / scenario_count if scenario_count > 0 else 0,
                    'num_affected_sectors': len(all_sector_impacts),
                    'sector_impacts': aggregated_impacts
                }

    return filtered_results


def filter_results_by_sectors(scenario_analyzer, sector_list):
    """Filter and aggregate results for specific sectors only."""
    filtered_results = {}

    # Get all effect types from the original results
    effect_types = list(scenario_analyzer.results.keys())

    for effect_type in effect_types:
        filtered_results[effect_type] = {}

        if effect_type not in scenario_analyzer.results:
            continue

        # Process each year
        for year, year_data in scenario_analyzer.results[effect_type].items():
            # Filter scenarios by sector
            filtered_year_data = {}

            for scenario_key, scenario_data in year_data.items():
                # Extract scenario index
                scenario_idx = int(scenario_key.split('_')[1])
                scenario_row = scenario_analyzer.scenarios_data.iloc[scenario_idx]

                # Check if this scenario matches our sector filter
                if str(scenario_row['sector']) in sector_list:
                    filtered_year_data[scenario_key] = scenario_data

            if filtered_year_data:
                # Aggregate the filtered results for this year
                all_sector_impacts = {}
                total_aggregate_impact = 0
                scenario_count = 0

                for scenario_key, scenario_data in filtered_year_data.items():
                    result = scenario_data['result']
                    total_aggregate_impact += result['total_impact']
                    scenario_count += 1

                    # Aggregate sector-level impacts
                    for impact in result['impacts']:
                        sector_code = str(impact['sector_code'])
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

                aggregated_impacts.sort(key=lambda x: abs(x['total_impact']), reverse=True)

                filtered_results[effect_type][year] = {
                    'total_aggregate_impact': total_aggregate_impact,
                    'scenario_count': scenario_count,
                    'avg_aggregate_impact': total_aggregate_impact / scenario_count if scenario_count > 0 else 0,
                    'num_affected_sectors': len(all_sector_impacts),
                    'sector_impacts': aggregated_impacts
                }

    return filtered_results


def show_integrated_tables():
    """Display integrated tables combining IO sectors 1610 and 4506."""
    st.subheader("🔗 Integrated Table Analysis")

    # Check if scenario analysis has been run
    if not st.session_state.get('scenario_results') or not st.session_state.get('scenario_analyzer'):
        st.warning("⚠️ No scenario analysis results available. Please run scenario analysis first.")
        st.info("👈 Go to the 'Run Analysis' menu to generate data.")
        return

    scenario_analyzer = st.session_state.scenario_analyzer

    # Check if scenario sheet names are available
    if not hasattr(scenario_analyzer, 'scenario_sheet_names'):
        st.error("Scenario sheet information not available.")
        return

    # Select scenario sheet
    scenario_sheets = scenario_analyzer.scenario_sheet_names
    selected_sheet = st.selectbox(
        "📋 Select Scenario Sheet:",
        options=scenario_sheets,
        key="integrated_scenario_selection",
        help="Choose which scenario sheet to analyze"
    )

    st.info(f"**Analyzing scenario sheet:** {selected_sheet}")
    st.markdown("---")

    # Get results for 1610 and 4506 combined
    results = {}

    # Get all years
    all_years_set = set()
    for effect_type in scenario_analyzer.aggregated_results.keys():
        if scenario_analyzer.aggregated_results[effect_type]:
            all_years_set.update(scenario_analyzer.aggregated_results[effect_type].keys())
    all_years = sorted(all_years_set)

    # Build combined results for 1610+4506
    effect_types_to_process = ['indirect_prod', 'indirect_import', 'value_added', 'jobcoeff', 'directemploycoeff']

    for effect_type in effect_types_to_process:
        results[effect_type] = {}
        for year in all_years:
            total_impact = 0
            all_sector_impacts = {}

            # Combine 1610 and 4506
            for sector_code in ['1610', '4506']:
                sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, selected_sheet, sector_code)

                if effect_type in sector_results and year in sector_results[effect_type]:
                    year_data = sector_results[effect_type][year]
                    total_impact += year_data['total_aggregate_impact']

                    # Merge sector impacts
                    for impact in year_data['sector_impacts']:
                        sector_key = str(impact['sector_code'])
                        if sector_key not in all_sector_impacts:
                            all_sector_impacts[sector_key] = {
                                'sector_code': impact['sector_code'],
                                'sector_name': impact['sector_name'],
                                'total_impact': 0
                            }
                        all_sector_impacts[sector_key]['total_impact'] += impact['total_impact']

            if total_impact != 0 or all_sector_impacts:
                # Convert to list and sort
                sector_impacts_list = [
                    {
                        'sector_code': v['sector_code'],
                        'sector_name': v['sector_name'],
                        'total_impact': v['total_impact'],
                        'avg_impact': v['total_impact'],  # Since we're combining, this is the total
                        'scenario_count': 1
                    }
                    for v in all_sector_impacts.values()
                ]
                sector_impacts_list.sort(key=lambda x: abs(x['total_impact']), reverse=True)

                results[effect_type][year] = {
                    'total_aggregate_impact': total_impact,
                    'scenario_count': 1,
                    'avg_aggregate_impact': total_impact,
                    'num_affected_sectors': len(all_sector_impacts),
                    'sector_impacts': sector_impacts_list
                }

    # SUMMARY TABLES SECTION
    st.markdown("### 📋 Summary Table (1610 + 4506)")

    summary_years = target_years

    # Define effect types to include in summary
    effect_columns = {
        'indirect_prod': 'Indirect Production Impact (Million KRW)',
        'indirect_import': 'Indirect Import Impact (Million KRW)',
        'value_added': 'Value Added Impact (Million KRW)',
        'jobcoeff': 'Job Creation Impact (Persons)'
    }

    # Build consolidated summary table
    summary_data = []
    for year in summary_years:
        row = {'Year': year}

        for effect_type, column_name in effect_columns.items():
            if effect_type in results and year in results[effect_type]:
                # Get total aggregate impact from the scenario_analyzer results
                total_impact = results[effect_type][year]['total_aggregate_impact']
                row[column_name] = total_impact
            else:
                row[column_name] = 0

        summary_data.append(row)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # Remove columns where all values are 0
        cols_to_keep = ['Year']
        for col in summary_df.columns:
            if col != 'Year':
                if summary_df[col].sum() != 0:  # Keep column if sum is not 0
                    cols_to_keep.append(col)
        summary_df = summary_df[cols_to_keep]

        # Format for display
        display_df = summary_df.copy()
        for col in display_df.columns:
            if col != 'Year':
                display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}" if x != 0 else "0.00")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # FULL TABLES SECTION
    st.markdown("### 📊 Full Tables")

    # Get ALL years from results
    all_years_set = set()
    for effect_type in results.keys():
        if results[effect_type]:
            all_years_set.update(results[effect_type].keys())
    all_years = sorted(all_years_set)

    # Get all effect types that have results
    available_effects = [effect for effect in results.keys() if results[effect]]

    if not available_effects:
        st.error("No effect types available in results.")
        return

    # Effect type descriptions
    effect_descriptions = {
        'indirect_prod': '💰 Indirect Production Effects',
        'indirect_import': '🌐 Indirect Import Effects',
        'value_added': '💎 Value Added Effects',
        'productioncoeff': '⚡ Production Coefficient (H2)',
        'valueaddedcoeff': '💎 Value Added Coefficient (H2)',
        'jobcoeff': '👥 Job Creation Effects',
        'directemploycoeff': '👔 Direct Employment Effects'
    }

    # Define all effect columns
    all_effect_columns = {
        'indirect_prod': 'Indirect Production (Million KRW)',
        'indirect_import': 'Indirect Import (Million KRW)',
        'value_added': 'Value Added (Million KRW)',
        'jobcoeff': 'Job Creation (Persons)',
        'directemploycoeff': 'Direct Employment (Persons)'
    }

    # Build consolidated full table

    full_data = []
    for year in all_years:
        row = {'Year': year}

        for effect_type, column_name in all_effect_columns.items():
            if effect_type in available_effects and year in results[effect_type]:
                # Get total aggregate impact from the scenario_analyzer results
                total_impact = results[effect_type][year]['total_aggregate_impact']
                row[column_name] = total_impact
            else:
                row[column_name] = 0

        full_data.append(row)

    if full_data:
        full_df = pd.DataFrame(full_data)

        # Format for display
        display_df = full_df.copy()
        for col in display_df.columns:
            if col != 'Year':
                display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}" if x != 0 else "0.00")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Download button (Excel)
        try:
            from io import BytesIO

            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                full_df.to_excel(writer, sheet_name='Consolidated Full Data', index=False)

            buffer.seek(0)

            st.download_button(
                label=f"📥 Download Consolidated Full Table (Excel)",
                data=buffer,
                file_name=f"integrated_full_consolidated_table.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_integrated_full_consolidated"
            )
        except ImportError:
            # Fallback to CSV
            csv = full_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label=f"📥 Download Consolidated Full Table (CSV)",
                data=csv,
                file_name=f"integrated_full_consolidated_table.csv",
                mime="text/csv",
                key=f"download_integrated_full_consolidated"
            )

    st.markdown("---")
    
    st.markdown("#### 📋 Detailed Analysis by Effect Type")

    # Create tabs for each available effect type
    tab_names = [effect_descriptions.get(effect, effect) for effect in available_effects]
    effect_tabs = st.tabs(tab_names)

    for i, effect_type in enumerate(available_effects):
        with effect_tabs[i]:
            st.subheader(f"{effect_descriptions.get(effect_type, effect_type)}")

            if not results[effect_type]:
                st.warning(f"No data available for {effect_type}")
                continue

            # Create sector impact matrix for ALL years
            st.markdown("#### 🎯 Sector Impacts by Year (All Years)")

            # Collect all unique sectors
            all_sectors = set()
            for year in all_years:
                if year in results[effect_type]:
                    for impact in results[effect_type][year]['sector_impacts']:
                        all_sectors.add(str(impact['sector_code']))

            if all_sectors:
                # Create sector matrix
                sector_matrix = []
                for sector_code in sorted(all_sectors):
                    row = {'Sector Code': sector_code, 'Sector Name': ''}

                    # Add code_h and product_h if IO analyzer available
                    if hasattr(scenario_analyzer, 'io_analyzer') and scenario_analyzer.io_analyzer:
                        if effect_type not in ['jobcoeff', 'directemploycoeff']:
                            code_h = scenario_analyzer.io_analyzer.basic_to_code_h.get(sector_code, '')
                            product_h = scenario_analyzer.io_analyzer.code_h_to_product_h.get(code_h, '') if code_h else ''
                            row['Code_H'] = code_h
                            row['Category_H'] = product_h

                    # Add impact values for each year (ALL YEARS)
                    for year in all_years:
                        if year in results[effect_type]:
                            year_data = results[effect_type][year]
                            impact_value = 0
                            sector_name = ''
                            
                            for impact in year_data['sector_impacts']:
                                if str(impact['sector_code']) == sector_code:
                                    impact_value = impact['total_impact']
                                    sector_name = impact['sector_name']
                                    break
                            
                            row[str(year)] = impact_value
                            if not row['Sector Name']:
                                row['Sector Name'] = sector_name
                        else:
                            row[str(year)] = 0
                    
                    sector_matrix.append(row)
                
                sector_df = pd.DataFrame(sector_matrix)

                # Format numeric columns for display
                display_df = sector_df.copy()
                for year in all_years:
                    if str(year) in display_df.columns:
                        display_df[str(year)] = display_df[str(year)].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)

                # Display with option to show top N sectors
                top_n = st.slider(
                    "Show top N sectors by total impact",
                    min_value=10,
                    max_value=len(sector_df),
                    value=min(20, len(sector_df)),
                    step=10,
                    key=f"slider_{effect_type}"
                )

                # Calculate total impact across years for sorting
                year_cols = [str(year) for year in all_years if str(year) in sector_df.columns]
                sector_df['total_across_years'] = sector_df[year_cols].sum(axis=1)
                sector_df = sector_df.sort_values('total_across_years', ascending=False)
                
                # Remove the sorting column from display
                display_df = sector_df.drop(columns=['total_across_years']).head(top_n)
                
                # Format display
                for year in target_years:
                    if str(year) in display_df.columns:
                        display_df[str(year)] = display_df[str(year)].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
                
                st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

                # Download button for detailed data (Excel)
                try:
                    from io import BytesIO

                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        # Export the sector data
                        export_df = sector_df.drop(columns=['total_across_years'])
                        export_df.to_excel(writer, sheet_name=effect_type[:30], index=False)

                    buffer.seek(0)

                    st.download_button(
                        label=f"📥 Download {effect_type} Detailed Sector Data (Excel)",
                        data=buffer,
                        file_name=f"integrated_sectors_{effect_type}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"download_detailed_{effect_type}"
                    )
                except ImportError:
                    # Fallback to CSV if openpyxl not available
                    csv_detailed = sector_df.drop(columns=['total_across_years']).to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label=f"📥 Download {effect_type} Detailed Sector Data (CSV)",
                        data=csv_detailed,
                        file_name=f"integrated_sectors_{effect_type}.csv",
                        mime="text/csv",
                        key=f"download_detailed_{effect_type}"
                    )
            else:
                st.info("No sector impact data available for the selected years.")

    st.markdown("---")
    st.info("💡 Tip: Go to 'Run Analysis' menu to run analysis with different scenario files.")


def show_hydrogen_analysis():
    """Display hydrogen table analysis results for H2S and H2T."""
    st.subheader("⚡ Hydrogen Table Analysis")

    # Check if scenario analysis has been run
    if not st.session_state.get('scenario_results') or not st.session_state.get('scenario_analyzer'):
        st.warning("⚠️ No scenario analysis results available. Please run scenario analysis first.")
        st.info("👈 Go to the 'Run Analysis' menu to generate data.")
        return

    scenario_analyzer = st.session_state.scenario_analyzer

    # Check if scenario sheet names are available
    if not hasattr(scenario_analyzer, 'scenario_sheet_names'):
        st.error("Scenario sheet information not available.")
        return

    # Select scenario sheet
    scenario_sheets = scenario_analyzer.scenario_sheet_names
    selected_sheet = st.selectbox(
        "📋 Select Scenario Sheet:",
        options=scenario_sheets,
        key="hydrogen_scenario_selection",
        help="Choose which scenario sheet to analyze"
    )

    st.info(f"**Analyzing scenario sheet:** {selected_sheet}")
    st.markdown("---")

    # Get results for H2S and H2T only
    results = {}

    # Get all years
    all_years_set = set()
    for effect_type in scenario_analyzer.aggregated_results.keys():
        if scenario_analyzer.aggregated_results[effect_type]:
            all_years_set.update(scenario_analyzer.aggregated_results[effect_type].keys())
    all_years = sorted(all_years_set)

    # Build combined results for H2S+H2T only
    hydrogen_effects = ['productioncoeff', 'valueaddedcoeff', 'jobcoeff', 'directemploycoeff']

    for effect_type in hydrogen_effects:
        results[effect_type] = {}
        for year in all_years:
            total_impact = 0
            all_sector_impacts = {}

            # Combine H2S and H2T only
            for sector_code in ['H2S', 'H2T']:
                sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, selected_sheet, sector_code)

                if effect_type in sector_results and year in sector_results[effect_type]:
                    year_data = sector_results[effect_type][year]
                    total_impact += year_data['total_aggregate_impact']

                    # Merge sector impacts
                    for impact in year_data['sector_impacts']:
                        sector_key = str(impact['sector_code'])
                        if sector_key not in all_sector_impacts:
                            all_sector_impacts[sector_key] = {
                                'sector_code': impact['sector_code'],
                                'sector_name': impact['sector_name'],
                                'total_impact': 0
                            }
                        all_sector_impacts[sector_key]['total_impact'] += impact['total_impact']

            if total_impact != 0 or all_sector_impacts:
                # Convert to list and sort
                sector_impacts_list = [
                    {
                        'sector_code': v['sector_code'],
                        'sector_name': v['sector_name'],
                        'total_impact': v['total_impact'],
                        'avg_impact': v['total_impact'],
                        'scenario_count': 1
                    }
                    for v in all_sector_impacts.values()
                ]
                sector_impacts_list.sort(key=lambda x: abs(x['total_impact']), reverse=True)

                results[effect_type][year] = {
                    'total_aggregate_impact': total_impact,
                    'scenario_count': 1,
                    'avg_aggregate_impact': total_impact,
                    'num_affected_sectors': len(all_sector_impacts),
                    'sector_impacts': sector_impacts_list
                }

    # Filter available effects
    available_h2_effects = [effect for effect in hydrogen_effects if effect in results and results[effect]]

    if not available_h2_effects:
        st.warning("No hydrogen analysis results available.")
        return

    # SUMMARY TABLES SECTION
    st.markdown("### 📋 Summary Tables (H2S + H2T)")

    summary_years = target_years

    # Define hydrogen effect columns
    h2_effect_columns = {
        'productioncoeff': 'Indirect production (Million KRW)',
        'valueaddedcoeff': 'Value-added creation (Million KRW)',
        'jobcoeff': 'Job Creation (Million KRW)',
        'directemploycoeff': 'Direct Employment (Persons)'
    }

    # Build consolidated summary table
    summary_data = []
    for year in summary_years:
        row = {'Year': year}

        for effect_type, column_name in h2_effect_columns.items():
            if effect_type in available_h2_effects and year in results[effect_type]:
                # Get total aggregate impact from the scenario_analyzer results
                total_impact = results[effect_type][year]['total_aggregate_impact']
                row[column_name] = total_impact
            else:
                row[column_name] = 0

        summary_data.append(row)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # Format for display
        display_df = summary_df.copy()
        for col in display_df.columns:
            if col != 'Year':
                display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}" if x != 0 else "0.00")

        st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.markdown("---")

    # FULL TABLES SECTION
    st.markdown("### 📊 Full Tables (All Years)")

    # Get all years
    all_years_set = set()
    for effect_type in available_h2_effects:
        if results[effect_type]:
            all_years_set.update(results[effect_type].keys())
    all_years = sorted(all_years_set)

    # Define hydrogen effect columns for full table
    h2_effect_columns = {
        'productioncoeff': 'Production Coefficient (Million KRW)',
        'valueaddedcoeff': 'Value Added Coefficient (Million KRW)',
        'jobcoeff': 'Job Creation (Million KRW)',
        'directemploycoeff': 'Direct Employment (Persons)'
    }

    # Build consolidated full table
    full_data = []
    for year in all_years:
        row = {'Year': year}

        for effect_type, column_name in h2_effect_columns.items():
            if effect_type in available_h2_effects and year in results[effect_type]:
                # Get total aggregate impact from the scenario_analyzer results
                total_impact = results[effect_type][year]['total_aggregate_impact']
                row[column_name] = total_impact
            else:
                row[column_name] = 0

        full_data.append(row)

    if full_data:
        full_df = pd.DataFrame(full_data)

        # Format for display
        display_df = full_df.copy()
        for col in display_df.columns:
            if col != 'Year':
                display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}" if x != 0 else "0.00")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Download button (Excel)
        try:
            from io import BytesIO

            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                full_df.to_excel(writer, sheet_name='H2 Full Data', index=False)

            buffer.seek(0)

            st.download_button(
                label=f"📥 Download H2 Full Table (Excel)",
                data=buffer,
                file_name=f"h2_full_consolidated_table.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_h2_full_consolidated"
            )
        except ImportError:
            # Fallback to CSV
            csv = full_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label=f"📥 Download H2 Full Table (CSV)",
                data=csv,
                file_name=f"h2_full_consolidated_table.csv",
                mime="text/csv",
                key=f"download_h2_full_consolidated"
            )


def filter_and_group_by_code_h(scenario_analyzer, sector_list):
    """Filter results by sectors and group by code_h categories."""
    filtered_results = {}

    # Get all effect types from the original results
    effect_types = list(scenario_analyzer.results.keys())

    for effect_type in effect_types:
        filtered_results[effect_type] = {}

        if effect_type not in scenario_analyzer.results:
            continue

        # Process each year
        for year, year_data in scenario_analyzer.results[effect_type].items():
            # Filter scenarios by sector
            filtered_year_data = {}

            for scenario_key, scenario_data in year_data.items():
                # Extract scenario index
                scenario_idx = int(scenario_key.split('_')[1])
                scenario_row = scenario_analyzer.scenarios_data.iloc[scenario_idx]

                # Check if this scenario matches our sector filter
                if str(scenario_row['sector']) in sector_list:
                    filtered_year_data[scenario_key] = scenario_data

            if filtered_year_data:
                # Aggregate the filtered results for this year, grouped by code_h
                all_code_h_impacts = {}
                total_aggregate_impact = 0
                scenario_count = 0

                for scenario_key, scenario_data in filtered_year_data.items():
                    result = scenario_data['result']
                    total_aggregate_impact += result['total_impact']
                    scenario_count += 1

                    # Aggregate sector-level impacts by code_h
                    for impact in result['impacts']:
                        sector_code = str(impact['sector_code'])
                        sector_name = impact['sector_name']
                        impact_value = impact['impact']

                        # Map to code_h
                        code_h = ''
                        product_h = ''
                        if hasattr(scenario_analyzer, 'io_analyzer') and scenario_analyzer.io_analyzer:
                            code_h = scenario_analyzer.io_analyzer.basic_to_code_h.get(sector_code, sector_code)
                            product_h = scenario_analyzer.io_analyzer.code_h_to_product_h.get(code_h, '') if code_h else ''

                        # Use code_h as the grouping key
                        group_key = code_h if code_h else sector_code

                        if group_key not in all_code_h_impacts:
                            all_code_h_impacts[group_key] = {
                                'code_h': code_h,
                                'product_h': product_h,
                                'total_impact': 0,
                                'scenario_count': 0
                            }

                        all_code_h_impacts[group_key]['total_impact'] += impact_value
                        all_code_h_impacts[group_key]['scenario_count'] += 1

                # Convert to sorted list
                aggregated_impacts = []
                for group_key, data in all_code_h_impacts.items():
                    aggregated_impacts.append({
                        'code_h': data['code_h'],
                        'product_h': data['product_h'],
                        'total_impact': data['total_impact'],
                        'avg_impact': data['total_impact'] / data['scenario_count'],
                        'scenario_count': data['scenario_count']
                    })

                aggregated_impacts.sort(key=lambda x: abs(x['total_impact']), reverse=True)

                filtered_results[effect_type][year] = {
                    'total_aggregate_impact': total_aggregate_impact,
                    'scenario_count': scenario_count,
                    'avg_aggregate_impact': total_aggregate_impact / scenario_count if scenario_count > 0 else 0,
                    'num_affected_categories': len(all_code_h_impacts),
                    'code_h_impacts': aggregated_impacts
                }

    return filtered_results


def show_total_tables():
    """Display total tables with 1610 + 4506 + H2 grouped by code_h categories."""
    st.subheader("📊 Total Table Summary (Coal+Renewable+H2)")

    # Check if scenario analysis has been run
    if not st.session_state.get('scenario_results') or not st.session_state.get('scenario_analyzer'):
        st.warning("⚠️ No scenario analysis results available. Please run scenario analysis first.")
        st.info("👈 Go to the 'Run Analysis' menu to generate data.")
        return

    scenario_analyzer = st.session_state.scenario_analyzer

    # Check if scenario sheet names are available
    if not hasattr(scenario_analyzer, 'scenario_sheet_names'):
        st.error("Scenario sheet information not available.")
        return

    # Select scenario sheet
    scenario_sheets = scenario_analyzer.scenario_sheet_names
    selected_sheet = st.selectbox(
        "📋 Select Scenario Sheet:",
        options=scenario_sheets,
        key="total_scenario_selection",
        help="Choose which scenario sheet to analyze"
    )

    st.info(f"**Analyzing scenario sheet:** {selected_sheet}")
    st.markdown("---")

    # Filter results by scenario sheet first
    sheet_results = filter_results_by_scenario_sheet(scenario_analyzer, selected_sheet)

    # Now group by code_h - need to adapt filter_and_group_by_code_h to work with already filtered results
    # For now, we'll use the sheet_results directly
    results = sheet_results

    # Get all effect types that have results
    available_effects = [effect for effect in results.keys() if results[effect]]

    if not available_effects:
        st.warning("No analysis results available.")
        return

    # SUMMARY TABLES SECTION
    st.markdown("### 📋 Summary Tables")
    st.markdown("Combined impacts: Indirect Production, Value Added, Job Creation")

    summary_years = target_years

    # Build consolidated summary table with combined effects
    summary_data = []
    for year in summary_years:
        row = {'Year': year}

        # 1. Indirect Production = indirect_prod + productioncoeff (H2)
        indirect_prod_val = 0
        if 'indirect_prod' in available_effects and year in results['indirect_prod']:
            indirect_prod_val += results['indirect_prod'][year]['total_aggregate_impact']
        if 'productioncoeff' in available_effects and year in results['productioncoeff']:
            indirect_prod_val += results['productioncoeff'][year]['total_aggregate_impact']
        row['Indirect Production (Million KRW)'] = indirect_prod_val

        # 2. Indirect Import (IO only)
        if 'indirect_import' in available_effects and year in results['indirect_import']:
            row['Indirect Import (Million KRW)'] = results['indirect_import'][year]['total_aggregate_impact']
        else:
            row['Indirect Import (Million KRW)'] = 0

        # 3. Value Added = value_added + valueaddedcoeff (H2)
        value_added_val = 0
        if 'value_added' in available_effects and year in results['value_added']:
            value_added_val += results['value_added'][year]['total_aggregate_impact']
        if 'valueaddedcoeff' in available_effects and year in results['valueaddedcoeff']:
            value_added_val += results['valueaddedcoeff'][year]['total_aggregate_impact']
        row['Value Added (Million KRW)'] = value_added_val

        # 4. Job Creation = jobcoeff for 1610+4506 + directemploycoeff for H2S+H2T
        job_creation_val = 0

        # Get jobcoeff from 1610 and 4506
        for sector_code in ['1610', '4506']:
            sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, selected_sheet, sector_code)
            if 'jobcoeff' in sector_results and year in sector_results['jobcoeff']:
                job_creation_val += sector_results['jobcoeff'][year]['total_aggregate_impact']

        # Get directemploycoeff from H2S and H2T
        for sector_code in ['H2S', 'H2T']:
            sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, selected_sheet, sector_code)
            if 'directemploycoeff' in sector_results and year in sector_results['directemploycoeff']:
                job_creation_val += sector_results['directemploycoeff'][year]['total_aggregate_impact']

        row['Job Creation (Persons)'] = job_creation_val

        summary_data.append(row)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # Remove columns where all values are 0 (except Job Creation - always show)
        cols_to_keep = ['Year']
        for col in summary_df.columns:
            if col != 'Year':
                # Always keep Job Creation column, even if zero
                if col == 'Job Creation (Persons)' or summary_df[col].sum() != 0:
                    cols_to_keep.append(col)
        summary_df = summary_df[cols_to_keep]

        # Format for display
        display_df = summary_df.copy()
        for col in display_df.columns:
            if col != 'Year':
                display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}" if x != 0 else "0.00")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # DETAILED TABLES BY CODE_H
    st.markdown("### 📊 Detailed Impact by Code_H Category")

    # Effect type descriptions
    effect_descriptions = {
        'indirect_prod': '💰 Indirect Production Effects',
        'indirect_import': '🌐 Indirect Import Effects',
        'value_added': '💎 Value Added Effects',
        'productioncoeff': '⚡ Production Coefficient (H2)',
        'valueaddedcoeff': '💎 Value Added Coefficient (H2)',
        'directemploycoeff': '👔 Direct Employment Effects'
    }

    # Create tabs for each available effect type
    tab_names = [effect_descriptions.get(effect, effect) for effect in available_effects]
    effect_tabs = st.tabs(tab_names)

    for i, effect_type in enumerate(available_effects):
        with effect_tabs[i]:
            st.subheader(f"{effect_descriptions.get(effect_type, effect_type)}")

            if not results[effect_type]:
                st.warning(f"No data available for {effect_type}")
                continue

            # Get all years
            all_years = sorted(results[effect_type].keys())

            # Create code_h impact matrix
            st.markdown("#### 📊 Impact by Code_H Category and Year")

            # Collect all unique code_h categories
            all_code_h = {}
            for year in all_years:
                if 'code_h_impacts' in results[effect_type][year]:
                    for impact in results[effect_type][year]['code_h_impacts']:
                        code_h = impact['code_h']
                        if code_h not in all_code_h:
                            all_code_h[code_h] = impact['product_h']

            if all_code_h:
                # Create matrix
                matrix_data = []
                for code_h, product_h in sorted(all_code_h.items()):
                    row = {
                        'Code_H': code_h,
                        'Category_H': product_h
                    }

                    # Add impact values for each year
                    for year in all_years:
                        impact_value = 0
                        if 'code_h_impacts' in results[effect_type][year]:
                            for impact in results[effect_type][year]['code_h_impacts']:
                                if impact['code_h'] == code_h:
                                    impact_value = impact['total_impact']
                                    break
                        row[str(year)] = impact_value

                    matrix_data.append(row)

                if matrix_data:
                    matrix_df = pd.DataFrame(matrix_data)

                    # Calculate total impact across years for sorting
                    year_cols = [str(year) for year in all_years]
                    matrix_df['total_across_years'] = matrix_df[year_cols].sum(axis=1)
                    matrix_df = matrix_df.sort_values('total_across_years', ascending=False)

                    # Format for display
                    display_df = matrix_df.drop(columns=['total_across_years']).copy()
                    for year_col in year_cols:
                        if year_col in display_df.columns:
                            display_df[year_col] = display_df[year_col].apply(lambda x: f"{x:,.2f}" if x != 0 else "0.00")

                    st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

                    # Download button
                    try:
                        from io import BytesIO

                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            export_df = matrix_df.drop(columns=['total_across_years'])
                            export_df.to_excel(writer, sheet_name=effect_type[:30], index=False)

                        buffer.seek(0)

                        st.download_button(
                            label=f"📥 Download {effect_type} by Code_H (Excel)",
                            data=buffer,
                            file_name=f"total_{effect_type}_by_code_h.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"download_total_{effect_type}"
                        )
                    except ImportError:
                        # Fallback to CSV
                        csv_data = matrix_df.drop(columns=['total_across_years']).to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label=f"📥 Download {effect_type} by Code_H (CSV)",
                            data=csv_data,
                            file_name=f"total_{effect_type}_by_code_h.csv",
                            mime="text/csv",
                            key=f"download_total_{effect_type}"
                        )
            else:
                st.info("No code_h category data available")

def show_scenario_comparison():
    """Display comparison between different scenario sheets by sector."""
    st.subheader("🔀 Scenario Sheet Comparison by Sector")

    # Check if scenario analysis has been run
    if not st.session_state.get('scenario_results') or not st.session_state.get('scenario_analyzer'):
        st.warning("⚠️ No scenario analysis results available. Please run scenario analysis first.")
        st.info("👈 Go to the 'Run Analysis' menu to generate data.")
        return

    scenario_analyzer = st.session_state.scenario_analyzer

    # Get scenario sheet names
    if not hasattr(scenario_analyzer, 'scenario_sheet_names'):
        st.error("Scenario sheet information not available.")
        return

    scenario_sheets = scenario_analyzer.scenario_sheet_names
    st.info(f"**Available scenario sheets:** {', '.join(scenario_sheets)}")

    st.markdown("---")

    # Select sector grouping
    sector_options = {
        '1610': '🏭 Sector 1610 (Coal)',
        '4506': '♻️ Sector 4506 (Renewable)',
        '1610+4506': '🔗 1610 + 4506 (Combined IO)',
        'H2S': '⚡ H2S (Hydrogen Storage)',
        'H2T': '🚛 H2T (Hydrogen Transport)',
        '1610+4506+H2S+H2T': '📊 Total (All Sectors)'
    }

    selected_sector_group = st.selectbox(
        "Select sector grouping:",
        options=list(sector_options.keys()),
        format_func=lambda x: sector_options[x],
        key="scenario_comparison_sector"
    )

    # Select effect type for comparison
    # Filter effect options based on sector grouping
    if selected_sector_group == '1610+4506+H2S+H2T':
        # For Total: exclude H2-specific coefficients (already included in indirect_prod and value_added)
        effect_options = {
            'indirect_prod': '💰 Indirect Production (IO + H2)',
            'indirect_import': '🌐 Indirect Import',
            'value_added': '💎 Value Added (IO + H2)',
            'directemploycoeff': '👔 Direct Employment'
        }
    elif selected_sector_group in ['H2S', 'H2T']:
        # For H2 sectors only: show H2-specific effects
        effect_options = {
            'productioncoeff': '⚡ Production Coefficient (H2)',
            'valueaddedcoeff': '💎 Value Added Coefficient (H2)',
            'jobcoeff': '👥 Job Creation',
            'directemploycoeff': '👔 Direct Employment'
        }
    else:
        # For IO sectors: show IO-specific effects
        effect_options = {
            'indirect_prod': '💰 Indirect Production',
            'indirect_import': '🌐 Indirect Import',
            'value_added': '💎 Value Added',
            'jobcoeff': '👥 Job Creation',
            'directemploycoeff': '👔 Direct Employment'
        }

    selected_effect = st.selectbox(
        "Select effect type to compare:",
        options=list(effect_options.keys()),
        format_func=lambda x: effect_options[x],
        key="scenario_comparison_effect"
    )

    st.markdown("---")
    st.markdown(f"### 📊 Comparison: {sector_options[selected_sector_group]} - {effect_options[selected_effect]}")

    # Get all years
    all_years_set = set()
    for effect_type in scenario_analyzer.aggregated_results.keys():
        if scenario_analyzer.aggregated_results[effect_type]:
            all_years_set.update(scenario_analyzer.aggregated_results[effect_type].keys())
    all_years = sorted(all_years_set)

    # Build comparison table based on selected sector grouping
    comparison_data = []

    for year in all_years:
        row = {'Year': year}

        for sheet_name in scenario_sheets:
            # Determine which sectors to include
            if selected_sector_group in ['1610', '4506', 'H2S', 'H2T']:
                # Single sector
                sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, sheet_name, selected_sector_group)

                if selected_effect in sector_results and year in sector_results[selected_effect]:
                    total_impact = sector_results[selected_effect][year]['total_aggregate_impact']
                    row[sheet_name] = total_impact
                else:
                    row[sheet_name] = 0

            elif selected_sector_group == '1610+4506':
                # Combined IO sectors
                total_impact = 0
                for sector_code in ['1610', '4506']:
                    sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, sheet_name, sector_code)
                    if selected_effect in sector_results and year in sector_results[selected_effect]:
                        total_impact += sector_results[selected_effect][year]['total_aggregate_impact']
                row[sheet_name] = total_impact

            elif selected_sector_group == '1610+4506+H2S+H2T':
                # All sectors combined
                if selected_effect == 'indirect_prod':
                    # Indirect Production = IO's indirect_prod + H2's productioncoeff
                    total_impact = 0
                    # Get IO sectors' indirect_prod
                    for sector_code in ['1610', '4506']:
                        sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, sheet_name, sector_code)
                        if 'indirect_prod' in sector_results and year in sector_results['indirect_prod']:
                            total_impact += sector_results['indirect_prod'][year]['total_aggregate_impact']
                    # Get H2 sectors' productioncoeff
                    for sector_code in ['H2S', 'H2T']:
                        sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, sheet_name, sector_code)
                        if 'productioncoeff' in sector_results and year in sector_results['productioncoeff']:
                            total_impact += sector_results['productioncoeff'][year]['total_aggregate_impact']
                    row[sheet_name] = total_impact
                elif selected_effect == 'value_added':
                    # Value Added = IO's value_added + H2's valueaddedcoeff
                    total_impact = 0
                    # Get IO sectors' value_added
                    for sector_code in ['1610', '4506']:
                        sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, sheet_name, sector_code)
                        if 'value_added' in sector_results and year in sector_results['value_added']:
                            total_impact += sector_results['value_added'][year]['total_aggregate_impact']
                    # Get H2 sectors' valueaddedcoeff
                    for sector_code in ['H2S', 'H2T']:
                        sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, sheet_name, sector_code)
                        if 'valueaddedcoeff' in sector_results and year in sector_results['valueaddedcoeff']:
                            total_impact += sector_results['valueaddedcoeff'][year]['total_aggregate_impact']
                    row[sheet_name] = total_impact
                else:
                    # Other effects: sum across all sectors
                    total_impact = 0
                    for sector_code in ['1610', '4506', 'H2S', 'H2T']:
                        sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, sheet_name, sector_code)
                        if selected_effect in sector_results and year in sector_results[selected_effect]:
                            total_impact += sector_results[selected_effect][year]['total_aggregate_impact']
                    row[sheet_name] = total_impact

        comparison_data.append(row)

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)

        # Format for display
        display_df = comparison_df.copy()
        for col in display_df.columns:
            if col != 'Year':
                display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}" if x != 0 else "0.00")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Download button
        try:
            from io import BytesIO

            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                comparison_df.to_excel(writer, sheet_name='Scenario Comparison', index=False)

            buffer.seek(0)

            st.download_button(
                label=f"📥 Download Comparison (Excel)",
                data=buffer,
                file_name=f"scenario_comparison_{selected_sector_group}_{selected_effect}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_scenario_comparison"
            )
        except ImportError:
            # Fallback to CSV
            csv_data = comparison_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label=f"📥 Download Comparison (CSV)",
                data=csv_data,
                file_name=f"scenario_comparison_{selected_sector_group}_{selected_effect}.csv",
                mime="text/csv",
                key=f"download_scenario_comparison"
            )

        # Visualization: Line chart comparing scenarios over years
        st.markdown("---")
        st.markdown("### 📈 Trend Comparison")

        fig = go.Figure()

        for sheet_name in scenario_sheets:
            y_values = [comparison_df[comparison_df['Year'] == year][sheet_name].values[0]
                       if len(comparison_df[comparison_df['Year'] == year]) > 0 else 0
                       for year in all_years]

            fig.add_trace(go.Scatter(
                x=all_years,
                y=y_values,
                mode='lines+markers',
                name=sheet_name,
                line=dict(width=2),
                marker=dict(size=8)
            ))

        fig.update_layout(
            title=f"{sector_options[selected_sector_group]} - {effect_options[selected_effect]}",
            xaxis_title="Year",
            yaxis_title="Impact Value",
            hovermode='x unified',
            height=500,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        st.plotly_chart(fig, use_container_width=True)


def show_individual_tables():
    """Display individual sector tables for 1610, 4506, H2S, H2T."""
    st.subheader("👤 Individual Sector Analysis")

    # Check if scenario analysis has been run
    if not st.session_state.get('scenario_results') or not st.session_state.get('scenario_analyzer'):
        st.warning("⚠️ No scenario analysis results available. Please run scenario analysis first.")
        st.info("👈 Go to the 'Run Analysis' menu to generate data.")
        return

    scenario_analyzer = st.session_state.scenario_analyzer

    # Check if scenario sheet names are available
    if not hasattr(scenario_analyzer, 'scenario_sheet_names'):
        st.error("Scenario sheet information not available.")
        return

    # Select scenario sheet
    scenario_sheets = scenario_analyzer.scenario_sheet_names
    selected_sheet = st.selectbox(
        "📋 Select Scenario Sheet:",
        options=scenario_sheets,
        key="individual_scenario_selection",
        help="Choose which scenario sheet to analyze"
    )

    st.info(f"**Analyzing scenario sheet:** {selected_sheet}")
    st.markdown("---")

    # Define sectors to analyze
    sectors = {
        '1610': '🏭 Sector 1610 (Coal)',
        '4506': '♻️ Sector 4506 (Renewable Energy)',
        'H2S': '⚡ H2S (Hydrogen Storage)',
        'H2T': '🚛 H2T (Hydrogen Transportation)'
    }

    # Create tabs for each sector
    sector_tabs = st.tabs([sectors[s] for s in sectors.keys()])

    for i, (sector_code, sector_name) in enumerate(sectors.items()):
        with sector_tabs[i]:
            st.markdown(f"### {sector_name}")

            # Filter results by scenario sheet and sector
            sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, selected_sheet, sector_code)

            # Check if this sector exists in the selected scenario sheet
            sector_exists = any(sector_results.get(effect, {}) for effect in sector_results.keys())

            if not sector_exists:
                st.warning(f"Sector {sector_code} not found in scenario sheet '{selected_sheet}'")
                continue

            # Get available effect types for this sector
            available_effects = [effect for effect in sector_results.keys() if sector_results[effect]]

            if not available_effects:
                st.warning(f"No analysis results available for sector {sector_code}.")
                continue

            # Build summary table
            summary_years = target_years

            # Define effect columns based on sector type
            if sector_code in ['H2S', 'H2T']:
                effect_columns = {
                    'productioncoeff': 'Indirect Production (Million KRW)',
                    'valueaddedcoeff': 'Value Added (Million KRW)',
                    'jobcoeff': 'Job Creation (Million KRW)',
                    'directemploycoeff': 'Direct Employment (Persons)'
                }
            else:  # 1610, 4506
                effect_columns = {
                    'indirect_prod': 'Indirect Production (Million KRW)',
                    'indirect_import': 'Indirect Import (Million KRW)',
                    'value_added': 'Value Added (Million KRW)',
                    'jobcoeff': 'Job Creation (Persons)',
                    'directemploycoeff': 'Direct Employment (Persons)'
                }

            # Build summary table
            summary_data = []
            for year in summary_years:
                row = {'Year': year}

                for effect_type, column_name in effect_columns.items():
                    if effect_type in available_effects and year in sector_results[effect_type]:
                        total_impact = sector_results[effect_type][year]['total_aggregate_impact']
                        row[column_name] = total_impact
                    else:
                        row[column_name] = 0

                summary_data.append(row)

            if summary_data:
                summary_df = pd.DataFrame(summary_data)

                # Remove columns where all values are 0
                cols_to_keep = ['Year']
                for col in summary_df.columns:
                    if col != 'Year':
                        if summary_df[col].sum() != 0:
                            cols_to_keep.append(col)
                summary_df = summary_df[cols_to_keep]

                # Format for display
                display_df = summary_df.copy()
                for col in display_df.columns:
                    if col != 'Year':
                        display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}" if x != 0 else "0.00")

                st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # FULL YEARS TABLE
            st.markdown("### 📊 Full Years Table")
            st.markdown("Impact data across all available years")

            # Get all years for this sector
            all_years_set = set()
            for effect_type in available_effects:
                if sector_results[effect_type]:
                    all_years_set.update(sector_results[effect_type].keys())
            all_years = sorted(all_years_set)

            # Build full years table
            full_data = []
            for year in all_years:
                row = {'Year': year}

                for effect_type, column_name in effect_columns.items():
                    if effect_type in available_effects and year in sector_results[effect_type]:
                        total_impact = sector_results[effect_type][year]['total_aggregate_impact']
                        row[column_name] = total_impact
                    else:
                        row[column_name] = 0

                full_data.append(row)

            if full_data:
                full_df = pd.DataFrame(full_data)

                # Remove columns where all values are 0
                cols_to_keep = ['Year']
                for col in full_df.columns:
                    if col != 'Year':
                        if full_df[col].sum() != 0:
                            cols_to_keep.append(col)
                full_df = full_df[cols_to_keep]

                # Format for display
                display_full_df = full_df.copy()
                for col in display_full_df.columns:
                    if col != 'Year':
                        display_full_df[col] = display_full_df[col].apply(lambda x: f"{x:,.2f}" if x != 0 else "0.00")

                st.dataframe(display_full_df, use_container_width=True, hide_index=True)

                # Download button
                try:
                    from io import BytesIO

                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        full_df.to_excel(writer, sheet_name=f'Sector {sector_code}', index=False)

                    buffer.seek(0)

                    st.download_button(
                        label=f"📥 Download {sector_code} Full Data (Excel)",
                        data=buffer,
                        file_name=f"sector_{sector_code}_full_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"download_sector_full_{sector_code}"
                    )
                except ImportError:
                    # Fallback to CSV
                    csv_data = full_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label=f"📥 Download {sector_code} Full Data (CSV)",
                        data=csv_data,
                        file_name=f"sector_{sector_code}_full_analysis.csv",
                        mime="text/csv",
                        key=f"download_sector_full_{sector_code}"
                    )

def show_summary_visualizations():
    """Display summary visualizations and charts."""
    st.title("📊 Visualizations")

    # Check if scenario analysis has been run
    if not st.session_state.get('scenario_results') or not st.session_state.get('scenario_analyzer'):
        st.warning("⚠️ No scenario analysis results available. Please run scenario analysis first.")
        st.info("👈 Go to the 'Run Analysis' menu to generate data.")
        return

    scenario_analyzer = st.session_state.scenario_analyzer

    # Import visualization module
    try:
        from libs.visualisation import Visualization
        viz = Visualization(scenario_analyzer)
    except Exception as e:
        st.error(f"Error loading visualization module: {e}")
        return

    # Create tabs for different visualization types
    viz_tabs = st.tabs(["📈 Yearly Trends", "🗺️ Sector Maps", "🔥 Code_H Heatmap", "📊 Scenario comparison"])

    # TAB 1: Yearly Trends
    with viz_tabs[0]:
        st.markdown("### 📈 Yearly Trends")
        st.markdown("Visualize how impacts change over time for different sectors and effect types")

        # Scenario sheet selector at the top
        if not hasattr(scenario_analyzer, 'scenario_sheet_names'):
            st.error("Scenario sheet information not available.")
        else:
            scenario_sheets = scenario_analyzer.scenario_sheet_names
            selected_sheet_trends = st.selectbox(
                "📋 Select Scenario Sheet:",
                options=scenario_sheets,
                key="trends_scenario_selection",
                help="Choose which scenario sheet to visualize"
            )

            st.info(f"**Visualizing scenario sheet:** {selected_sheet_trends}")
            st.markdown("---")

            # Sub-tabs for IO and Hydrogen
            trend_sub_tabs = st.tabs(["IO Table Trends", "Hydrogen Trends"])

            # IO Table Trends
            with trend_sub_tabs[0]:
                st.markdown("#### IO Table Yearly Trends (1610 + 4506)")

                io_effect = st.selectbox(
                    "Select Effect Type",
                    options=['indirect_prod', 'indirect_import', 'value_added', 'jobcoeff', 'directemploycoeff'],
                    format_func=lambda x: {
                        'indirect_prod': '💰 Indirect Production',
                        'indirect_import': '🌐 Indirect Import',
                        'value_added': '💎 Value Added',
                        'jobcoeff': '👥 Job Creation',
                        'directemploycoeff': '👔 Direct Employment'
                    }[x],
                    key="io_trend_effect"
                )

                sectors_io = st.multiselect(
                    "Select Sectors",
                    options=['1610', '4506', '1610+4506'],
                    default=['1610', '4506', '1610+4506'],
                    key="io_trend_sectors"
                )

                if st.button("Generate IO Trends", key="btn_io_trends"):
                    try:
                        # Get all years
                        all_years_set = set()
                        for effect_type in scenario_analyzer.aggregated_results.keys():
                            if scenario_analyzer.aggregated_results[effect_type]:
                                all_years_set.update(scenario_analyzer.aggregated_results[effect_type].keys())
                        all_years = sorted(all_years_set)

                        # Create figure
                        fig = go.Figure()

                        for sector_option in sectors_io:
                            if sector_option == '1610+4506':
                                # Combined sectors
                                y_values = []
                                for year in all_years:
                                    total = 0
                                    for sector in ['1610', '4506']:
                                        sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, selected_sheet_trends, sector)
                                        if io_effect in sector_results and year in sector_results[io_effect]:
                                            total += sector_results[io_effect][year]['total_aggregate_impact']
                                    y_values.append(total)

                                fig.add_trace(go.Scatter(
                                    x=all_years,
                                    y=y_values,
                                    mode='lines+markers',
                                    name='1610+4506',
                                    line=dict(width=3),
                                    marker=dict(size=8)
                                ))
                            else:
                                # Single sector
                                sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, selected_sheet_trends, sector_option)
                                y_values = []
                                for year in all_years:
                                    if io_effect in sector_results and year in sector_results[io_effect]:
                                        y_values.append(sector_results[io_effect][year]['total_aggregate_impact'])
                                    else:
                                        y_values.append(0)

                                fig.add_trace(go.Scatter(
                                    x=all_years,
                                    y=y_values,
                                    mode='lines+markers',
                                    name=sector_option,
                                    line=dict(width=2),
                                    marker=dict(size=7)
                                ))

                        fig.update_layout(
                            title=f"IO Table Trends - {selected_sheet_trends}",
                            xaxis_title="Year",
                            yaxis_title="Impact Value",
                            hovermode='x unified',
                            height=500
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating IO trends: {e}")

            # Hydrogen Trends
            with trend_sub_tabs[1]:
                st.markdown("#### Hydrogen Yearly Trends (H2S + H2T)")

                h2_effect = st.selectbox(
                    "Select Effect Type",
                    options=['productioncoeff', 'valueaddedcoeff', 'jobcoeff', 'directemploycoeff'],
                    format_func=lambda x: {
                        'productioncoeff': '⚡ Indirect Production (H2)',
                        'valueaddedcoeff': '💎 Value Added (H2)',
                        'jobcoeff': '👥 Job Creation',
                        'directemploycoeff': '👔 Direct Employment'
                    }[x],
                    key="h2_trend_effect"
                )

                sectors_h2 = st.multiselect(
                    "Select Scenarios",
                    options=['H2S', 'H2T', 'H2S+H2T'],
                    default=['H2S', 'H2T', 'H2S+H2T'],
                    key="h2_trend_sectors"
                )

                if st.button("Generate H2 Trends", key="btn_h2_trends"):
                    try:
                        # Get all years
                        all_years_set = set()
                        for effect_type in scenario_analyzer.aggregated_results.keys():
                            if scenario_analyzer.aggregated_results[effect_type]:
                                all_years_set.update(scenario_analyzer.aggregated_results[effect_type].keys())
                        all_years = sorted(all_years_set)

                        # Create figure
                        fig = go.Figure()

                        for sector_option in sectors_h2:
                            if sector_option == 'H2S+H2T':
                                # Combined sectors
                                y_values = []
                                for year in all_years:
                                    total = 0
                                    for sector in ['H2S', 'H2T']:
                                        sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, selected_sheet_trends, sector)
                                        if h2_effect in sector_results and year in sector_results[h2_effect]:
                                            total += sector_results[h2_effect][year]['total_aggregate_impact']
                                    y_values.append(total)

                                fig.add_trace(go.Scatter(
                                    x=all_years,
                                    y=y_values,
                                    mode='lines+markers',
                                    name='H2S+H2T',
                                    line=dict(width=3),
                                    marker=dict(size=8)
                                ))
                            else:
                                # Single sector
                                sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, selected_sheet_trends, sector_option)
                                y_values = []
                                for year in all_years:
                                    if h2_effect in sector_results and year in sector_results[h2_effect]:
                                        y_values.append(sector_results[h2_effect][year]['total_aggregate_impact'])
                                    else:
                                        y_values.append(0)

                                fig.add_trace(go.Scatter(
                                    x=all_years,
                                    y=y_values,
                                    mode='lines+markers',
                                    name=sector_option,
                                    line=dict(width=2),
                                    marker=dict(size=7)
                                ))

                        fig.update_layout(
                            title=f"Hydrogen Trends - {selected_sheet_trends}",
                            xaxis_title="Year",
                            yaxis_title="Impact Value",
                            hovermode='x unified',
                            height=500
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating H2 trends: {e}")

    # TAB 2: Sector Maps
    with viz_tabs[1]:
        st.markdown("### 🗺️ Top Sectors Analysis")
        st.markdown("Visualize which sectors have the highest impacts for a given year and effect type")

        # Scenario sheet selector at the top
        if not hasattr(scenario_analyzer, 'scenario_sheet_names'):
            st.error("Scenario sheet information not available.")
        else:
            scenario_sheets = scenario_analyzer.scenario_sheet_names
            selected_sheet_sectors = st.selectbox(
                "📋 Select Scenario Sheet:",
                options=scenario_sheets,
                key="sectors_scenario_selection",
                help="Choose which scenario sheet to visualize"
            )

            st.info(f"**Visualizing scenario sheet:** {selected_sheet_sectors}")
            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                sector_option = st.selectbox(
                    "Select Sector",
                    options=['1610', '4506', '1610+4506', 'H2S', 'H2T', 'H2S+H2T'],
                    format_func=lambda x: {
                        '1610': '🏭 Sector 1610 (Coal)',
                        '4506': '♻️ Sector 4506 (Renewable)',
                        '1610+4506': '🔗 1610 + 4506 (Combined)',
                        'H2S': '⚡ H2S (Hydrogen Storage)',
                        'H2T': '🚛 H2T (Hydrogen Transport)',
                        'H2S+H2T': '⚡ H2S + H2T (Combined)'
                    }[x],
                    key="sector_scenario"
                )

            with col2:
                sector_year = st.selectbox(
                    "Select Year",
                    options=[2026, 2030, 2040, 2050],
                    index=3,  # Default to 2050
                    key="sector_year"
                )

            # Effect type selection based on sector
            if sector_option in ['H2S', 'H2T', 'H2S+H2T']:
                effect_options = ['productioncoeff', 'valueaddedcoeff', 'jobcoeff', 'directemploycoeff']
                effect_labels = {
                    'productioncoeff': '⚡ Indirect Production (H2)',
                    'valueaddedcoeff': '💎 Value Added (H2)',
                    'jobcoeff': '👥 Job Creation',
                    'directemploycoeff': '👔 Direct Employment'
                }
            else:
                effect_options = ['indirect_prod', 'indirect_import', 'value_added', 'jobcoeff', 'directemploycoeff']
                effect_labels = {
                    'indirect_prod': '💰 Indirect Production',
                    'indirect_import': '🌐 Indirect Import',
                    'value_added': '💎 Value Added',
                    'jobcoeff': '👥 Job Creation',
                    'directemploycoeff': '👔 Direct Employment'
                }

            sector_effect = st.selectbox(
                "Select Effect Type",
                options=effect_options,
                format_func=lambda x: effect_labels[x],
                key="sector_effect"
            )

            if st.button("Generate Top 10 Sectors", key="btn_top10"):
                try:
                    # Get sector results based on selection
                    if sector_option in ['1610', '4506', 'H2S', 'H2T']:
                        # Single sector
                        sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, selected_sheet_sectors, sector_option)
                    elif sector_option == '1610+4506':
                        # Combined IO sectors - need to merge
                        sector_results = {}
                        for sector in ['1610', '4506']:
                            sr = filter_results_by_sheet_and_sector(scenario_analyzer, selected_sheet_sectors, sector)
                            for effect_type, year_data in sr.items():
                                if effect_type not in sector_results:
                                    sector_results[effect_type] = {}
                                for year, data in year_data.items():
                                    if year not in sector_results[effect_type]:
                                        sector_results[effect_type][year] = {
                                            'sector_impacts': [],
                                            'total_aggregate_impact': 0
                                        }
                                    sector_results[effect_type][year]['sector_impacts'].extend(data['sector_impacts'])
                                    sector_results[effect_type][year]['total_aggregate_impact'] += data['total_aggregate_impact']
                    elif sector_option == 'H2S+H2T':
                        # Combined H2 sectors
                        sector_results = {}
                        for sector in ['H2S', 'H2T']:
                            sr = filter_results_by_sheet_and_sector(scenario_analyzer, selected_sheet_sectors, sector)
                            for effect_type, year_data in sr.items():
                                if effect_type not in sector_results:
                                    sector_results[effect_type] = {}
                                for year, data in year_data.items():
                                    if year not in sector_results[effect_type]:
                                        sector_results[effect_type][year] = {
                                            'sector_impacts': [],
                                            'total_aggregate_impact': 0
                                        }
                                    sector_results[effect_type][year]['sector_impacts'].extend(data['sector_impacts'])
                                    sector_results[effect_type][year]['total_aggregate_impact'] += data['total_aggregate_impact']

                    # Get top 10 sectors for the selected year and effect
                    if sector_effect in sector_results and sector_year in sector_results[sector_effect]:
                        impacts = sector_results[sector_effect][sector_year]['sector_impacts']

                        # Sort by absolute impact and get top 10
                        top_10 = sorted(impacts, key=lambda x: abs(x['total_impact']), reverse=True)[:10]

                        # Create bar chart
                        fig = go.Figure()

                        sector_codes = [imp['sector_code'] for imp in top_10]
                        sector_names = [imp['sector_name'][:30] for imp in top_10]  # Truncate long names
                        impact_values = [imp['total_impact'] for imp in top_10]

                        fig.add_trace(go.Bar(
                            y=sector_names,
                            x=impact_values,
                            orientation='h',
                            marker=dict(color='#1f77b4'),
                            text=[f"{val:,.0f}" for val in impact_values],
                            textposition='auto'
                        ))

                        fig.update_layout(
                            title=f"Top 10 Sectors - {sector_option} - {selected_sheet_sectors}<br>{effect_labels[sector_effect]} ({sector_year})",
                            xaxis_title="Impact Value",
                            yaxis_title="Sector",
                            height=500,
                            yaxis=dict(autorange="reversed")
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No data available for the selected combination.")
                except Exception as e:
                    st.error(f"Error generating sector map: {e}")
                    st.exception(e)

    # TAB 3: Code_H Heatmap
    with viz_tabs[2]:
        st.markdown("### 🔥 Code_H Sector Heatmap")
        st.markdown("Interactive heatmap showing top sectors by Product_H category, ranked by impact magnitude")

        # Scenario sheet selector at the top
        if not hasattr(scenario_analyzer, 'scenario_sheet_names'):
            st.error("Scenario sheet information not available.")
        else:
            scenario_sheets = scenario_analyzer.scenario_sheet_names
            selected_sheet_heatmap = st.selectbox(
                "📋 Select Scenario Sheet:",
                options=scenario_sheets,
                key="heatmap_scenario_selection",
                help="Choose which scenario sheet to visualize"
            )

            st.info(f"**Visualizing scenario sheet:** {selected_sheet_heatmap}")
            st.markdown("---")

            # Get filtered results for the selected scenario sheet
            sheet_results = filter_results_by_scenario_sheet(scenario_analyzer, selected_sheet_heatmap)

            effect_labels = {
                'indirect_prod': '💰 Indirect Production',
                'indirect_import': '🌐 Indirect Import',
                'value_added': '💎 Value Added',
                'jobcoeff': '👥 Job Creation',
                'directemploycoeff': '👔 Direct Employment',
                'productioncoeff': '⚡ Production Coeff (H2)',
                'valueaddedcoeff': '💎 Value Added Coeff (H2)'
            }

            available_effects = [
                effect for effect in effect_labels.keys()
                if effect in sheet_results and sheet_results[effect]
            ]

            if not available_effects:
                st.warning("No effect types available for the selected scenario sheet.")
            else:
                st.info("💡 **How to use**: Choose an effect type and year, then generate the heatmap to view the top sectors per Product_H category.")

                col1, col2, col3 = st.columns(3)

                with col1:
                    heatmap_effect = st.selectbox(
                        "Effect Type",
                        options=available_effects,
                        format_func=lambda x: effect_labels[x],
                        key="heatmap_effect"
                    )

                available_years = sorted(sheet_results[heatmap_effect].keys())

                if not available_years:
                    st.warning("No yearly data available for the selected effect type.")
                else:
                    default_year_index = available_years.index(2030) if 2030 in available_years else 0

                    with col2:
                        heatmap_year = st.selectbox(
                            "Year",
                            options=available_years,
                            index=default_year_index,
                            key="heatmap_year"
                        )

                    with col3:
                        heatmap_top_n = st.slider(
                            "Top N Sectors per Category",
                            min_value=5,
                            max_value=20,
                            value=10,
                            step=5,
                            key="heatmap_top_n"
                        )

                    if st.button("🎨 Generate Heatmap", type="primary", key="btn_heatmap"):
                        with st.spinner("Creating interactive heatmap..."):
                            try:
                                year_data = sheet_results[heatmap_effect].get(heatmap_year)
                                if not year_data:
                                    st.error(f"No data found for {heatmap_year}.")
                                else:
                                    sector_impacts = year_data['sector_impacts']
                                    if not sector_impacts:
                                        st.warning("No sector impacts available for visualization.")
                                    else:
                                        data_rows = []
                                        value_column = f"{heatmap_effect}_{heatmap_year}"

                                        for impact in sector_impacts:
                                            sector_code = str(impact['sector_code'])
                                            sector_name = impact['sector_name']
                                            total_impact = impact['total_impact']

                                            code_h = ''
                                            product_h = ''
                                            if hasattr(scenario_analyzer, 'io_analyzer') and scenario_analyzer.io_analyzer:
                                                code_h = scenario_analyzer.io_analyzer.basic_to_code_h.get(sector_code, '')
                                                if code_h:
                                                    product_h = scenario_analyzer.io_analyzer.code_h_to_product_h.get(code_h, code_h)

                                            if not code_h:
                                                code_h = sector_code
                                                product_h = f"H2 Scenario ({sector_code})"

                                            data_rows.append({
                                                'Sector_Code': sector_code,
                                                'Sector_Name': sector_name,
                                                'Code_H': code_h,
                                                'Product_H': product_h,
                                                value_column: total_impact
                                            })

                                        if not data_rows:
                                            st.warning("Unable to assemble data for the heatmap.")
                                        else:
                                            df = pd.DataFrame(data_rows)

                                            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                                                output_path = tmp_file.name

                                            fig = Visualization.plot_code_h_sector_top10_heatmap_plotly(
                                                df=df,
                                                effect=heatmap_effect,
                                                year=heatmap_year,
                                                top_n=heatmap_top_n,
                                                output_path=output_path
                                            )

                                            try:
                                                os.remove(output_path)
                                            except OSError:
                                                pass

                                            st.plotly_chart(fig, use_container_width=True)
                                            st.success(f"✅ Heatmap generated successfully for {heatmap_year}!")

                            except Exception as e:
                                st.error(f"❌ Error generating heatmap: {e}")
                                st.exception(e)

    # TAB 4: Grid Comparison
    with viz_tabs[3]:
        st.markdown("### 📊 Multi-Scenario Comparison")
        st.markdown("Compare multiple effect types and scenarios in a single chart")

        # Check if scenario analyzer is available
        if not hasattr(scenario_analyzer, 'scenario_sheet_names'):
            st.error("Scenario sheet information not available.")
        else:
            scenario_sheets = scenario_analyzer.scenario_sheet_names

            # Select sector grouping
            sector_options_grid = {
                '1610': '🏭 Sector 1610 (Coal)',
                '4506': '♻️ Sector 4506 (Renewable)',
                '1610+4506': '🔗 1610 + 4506 (Combined IO)',
                'H2S': '⚡ H2S (Hydrogen Storage)',
                'H2T': '🚛 H2T (Hydrogen Transport)',
                '1610+4506+H2S+H2T': '📊 Total (All Sectors)'
            }

            selected_sector_grid = st.selectbox(
                "Select sector grouping:",
                options=list(sector_options_grid.keys()),
                format_func=lambda x: sector_options_grid[x],
                key="grid_sector_selection"
            )

            # Select scenarios to compare
            selected_scenarios_grid = st.multiselect(
                "Select scenario sheets to display:",
                options=scenario_sheets,
                default=scenario_sheets[:min(4, len(scenario_sheets))],  # Default to first 4
                key="grid_scenario_selection"
            )

            if len(selected_scenarios_grid) < 1:
                st.warning("Please select at least one scenario sheet.")
            else:
                # Select effect types to display based on sector grouping
                if selected_sector_grid == '1610+4506+H2S+H2T':
                    effect_options_grid = {
                        'indirect_prod': '💰 Indirect Production (IO + H2)',
                        'indirect_import': '🌐 Indirect Import',
                        'value_added': '💎 Value Added (IO + H2)',
                        'directemploycoeff': '👔 Direct Employment'
                    }
                elif selected_sector_grid in ['H2S', 'H2T']:
                    effect_options_grid = {
                        'productioncoeff': '⚡ Production Coefficient (H2)',
                        'valueaddedcoeff': '💎 Value Added Coefficient (H2)',
                        'jobcoeff': '👥 Job Creation',
                        'directemploycoeff': '👔 Direct Employment'
                    }
                else:
                    effect_options_grid = {
                        'indirect_prod': '💰 Indirect Production',
                        'indirect_import': '🌐 Indirect Import',
                        'value_added': '💎 Value Added',
                        'jobcoeff': '👥 Job Creation',
                        'directemploycoeff': '👔 Direct Employment'
                    }

                selected_effects_grid = st.multiselect(
                    "Select effect types to display:",
                    options=list(effect_options_grid.keys()),
                    default=list(effect_options_grid.keys())[:2],
                    format_func=lambda x: effect_options_grid[x],
                    key="grid_effect_selection"
                )

                if len(selected_effects_grid) < 1:
                    st.warning("Please select at least one effect type.")
                else:
                    # Generate grid button
                    if st.button("🎨 Generate Comparison Chart", type="primary", key="btn_grid_comparison"):
                        with st.spinner("Creating comparison chart..."):
                            # Get all years
                            all_years_set = set()
                            for effect_type in scenario_analyzer.aggregated_results.keys():
                                if scenario_analyzer.aggregated_results[effect_type]:
                                    all_years_set.update(scenario_analyzer.aggregated_results[effect_type].keys())
                            all_years = sorted(all_years_set)

                            # Create single figure with all combinations
                            fig = go.Figure()

                            # Color palette for different combinations
                            colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set2 + px.colors.qualitative.Pastel

                            line_idx = 0
                            for effect in selected_effects_grid:
                                for scenario in selected_scenarios_grid:
                                    y_values = []
                                    for year in all_years:
                                        # Calculate based on selected sector grouping
                                        if selected_sector_grid in ['1610', '4506', 'H2S', 'H2T']:
                                            # Single sector
                                            sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, scenario, selected_sector_grid)
                                            if effect in sector_results and year in sector_results[effect]:
                                                y_values.append(sector_results[effect][year]['total_aggregate_impact'])
                                            else:
                                                y_values.append(0)
                                        elif selected_sector_grid == '1610+4506':
                                            # Combined IO sectors
                                            total = 0
                                            for sector_code in ['1610', '4506']:
                                                sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, scenario, sector_code)
                                                if effect in sector_results and year in sector_results[effect]:
                                                    total += sector_results[effect][year]['total_aggregate_impact']
                                            y_values.append(total)
                                        elif selected_sector_grid == '1610+4506+H2S+H2T':
                                            # All sectors combined
                                            if effect == 'indirect_prod':
                                                # Indirect Production = IO's indirect_prod + H2's productioncoeff
                                                total = 0
                                                for sector_code in ['1610', '4506']:
                                                    sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, scenario, sector_code)
                                                    if 'indirect_prod' in sector_results and year in sector_results['indirect_prod']:
                                                        total += sector_results['indirect_prod'][year]['total_aggregate_impact']
                                                for sector_code in ['H2S', 'H2T']:
                                                    sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, scenario, sector_code)
                                                    if 'productioncoeff' in sector_results and year in sector_results['productioncoeff']:
                                                        total += sector_results['productioncoeff'][year]['total_aggregate_impact']
                                                y_values.append(total)
                                            elif effect == 'value_added':
                                                # Value Added = IO's value_added + H2's valueaddedcoeff
                                                total = 0
                                                for sector_code in ['1610', '4506']:
                                                    sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, scenario, sector_code)
                                                    if 'value_added' in sector_results and year in sector_results['value_added']:
                                                        total += sector_results['value_added'][year]['total_aggregate_impact']
                                                for sector_code in ['H2S', 'H2T']:
                                                    sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, scenario, sector_code)
                                                    if 'valueaddedcoeff' in sector_results and year in sector_results['valueaddedcoeff']:
                                                        total += sector_results['valueaddedcoeff'][year]['total_aggregate_impact']
                                                y_values.append(total)
                                            else:
                                                # Other effects: sum across all sectors
                                                total = 0
                                                for sector_code in ['1610', '4506', 'H2S', 'H2T']:
                                                    sector_results = filter_results_by_sheet_and_sector(scenario_analyzer, scenario, sector_code)
                                                    if effect in sector_results and year in sector_results[effect]:
                                                        total += sector_results[effect][year]['total_aggregate_impact']
                                                y_values.append(total)

                                    # Create line name
                                    line_name = f"{effect_options_grid[effect]} - {scenario}"

                                    # Different line styles for different effects
                                    line_dash = 'solid'
                                    if effect == 'indirect_import':
                                        line_dash = 'dash'
                                    elif effect == 'value_added':
                                        line_dash = 'dot'
                                    elif effect == 'jobcoeff':
                                        line_dash = 'dashdot'
                                    elif effect == 'directemploycoeff':
                                        line_dash = 'longdash'

                                    fig.add_trace(go.Scatter(
                                        x=all_years,
                                        y=y_values,
                                        mode='lines+markers',
                                        name=line_name,
                                        line=dict(width=2, dash=line_dash, color=colors[line_idx % len(colors)]),
                                        marker=dict(size=7),
                                        hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Impact: %{y:,.2f}<extra></extra>'
                                    ))

                                    line_idx += 1

                            # Update layout
                            fig.update_layout(
                                title="Multi-Scenario Effect Comparison",
                                xaxis_title="Year",
                                yaxis_title="Impact Value",
                                hovermode='x unified',
                                height=600,
                                legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=1.01,
                                    bgcolor="rgba(255, 255, 255, 0.8)"
                                ),
                                margin=dict(r=250)  # Extra margin for legend
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            st.success("✅ Comparison chart generated successfully!")

                            st.info("💡 **Tip**: Different line styles represent different effect types. Hover over lines to see details.")

def main():
    # Sidebar - Show scenario file info
    st.sidebar.title("🏭 Input Output Analysis")

    # Display currently loaded scenario file
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Data File")

    data_folder = Path("data")
    data_file = data_folder / "Data_v10.xlsx"

    if 'current_scenario_file' in st.session_state:
        current_file = st.session_state.current_scenario_file
        st.sidebar.success(f"✅ {current_file}")

        # Show loaded scenario sheets
        if 'scenario_analyzer' in st.session_state:
            analyzer = st.session_state.scenario_analyzer
            if hasattr(analyzer, 'scenario_sheet_names'):
                with st.sidebar.expander("📋 Loaded Scenario Sheets"):
                    for sheet_name in analyzer.scenario_sheet_names:
                        st.text(f"• {sheet_name}")
    else:
        st.sidebar.warning("⚠️ No data loaded")
        st.sidebar.info("Go to 'Run Analysis' menu to load Data_v10.xlsx")

    # Show data file status
    if data_file.exists():
        st.sidebar.markdown(f"**Data file:** `Data_v10.xlsx` ✅")
    else:
        st.sidebar.error("❌ Data_v10.xlsx not found")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📍 Main Menu")
    main_option = st.sidebar.radio(
        "      ",
        ["Run Analysis", "Table results", "Visualisation"],
        index=0
    )

    # Show the respective content area depending on high-level selection
    if main_option == "Run Analysis":
        # Run scenario analysis
        run_scenario_analysis()
    elif main_option == "Table results":
        # Show tabs for different table types
        st.title("📊 Analysis results")

        # Create tabs for different table views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔀 Scenario Comparison", "🔗 Integrated", "⚡ H2", "📊 Total", "👤 Individual"])

        with tab1:
            show_scenario_comparison()

        with tab2:
            show_integrated_tables()

        with tab3:
            show_hydrogen_analysis()

        with tab4:
            show_total_tables()

        with tab5:
            show_individual_tables()
    else:  # Visualisation
        show_summary_visualizations()


if __name__ == "__main__":
    main()