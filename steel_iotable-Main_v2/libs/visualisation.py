"""
Visualization Module for Steel I-O Table Analysis

Leverages existing methods from HydrogenTableAnalyzer, ScenarioAnalyzer, and IOTableAnalyzer.
"""

import plotly.graph_objects as go
import plotly.io as pio
from typing import List
from codecs import utf_8_decode
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# Set default Plotly template
pio.templates.default = "plotly_white"


class Visualization:
    """Visualization class using existing analyzer methods."""

    def __init__(self, scenario_analyzer):
        """
        Initialize with a ScenarioAnalyzer that has already run analysis.

        Args:
            scenario_analyzer: ScenarioAnalyzer instance with results
        """
        self.scenario_analyzer = scenario_analyzer

    def create_io_yearly_trends(self, effect_type: str, scenarios: List[str] = None,
                                show_fig: bool = True):
        """
        Create yearly trends for IO table scenarios.
        Uses scenario_analyzer.integrate_sectors_1610_4506() for integrated data.

        Args:
            effect_type: 'indirect_prod', 'indirect_import', 'value_added', 'jobcoeff', 'directemploycoeff'
            scenarios: List of scenarios (default: ['1610', '4506', '1610&4506'])
            show_fig: Whether to display the figure

        Returns:
            Plotly Figure
        """
        if scenarios is None:
            scenarios = ['1610', '4506', '1610&4506']

        effect_info = {
            'indirect_prod': {'label': 'Indirect Production', 'unit': 'Billion Won'},
            'indirect_import': {'label': 'Import', 'unit': 'Billion Won'},
            'value_added': {'label': 'Value Added', 'unit': 'Billion Won'},
            'jobcoeff': {'label': 'Job Creation', 'unit': 'Person'},
            'directemploycoeff': {'label': 'Direct Employment', 'unit': 'Person'}
        }

        fig = go.Figure()

        # Get integrated results if needed
        integrated_io = None
        if '1610&4506' in scenarios:
            integrated_io = self.scenario_analyzer.integrate_sectors_1610_4506()

        for scenario in scenarios:
            if scenario == '1610&4506' and integrated_io:
                # Use integration method results
                if effect_type in integrated_io:
                    years = sorted(integrated_io[effect_type].keys())
                    # Don't divide job creation by 1000 - keep as persons
                    divisor = 1000 if effect_type in ['indirect_prod', 'indirect_import', 'value_added'] else 1
                    values = [integrated_io[effect_type][y]['total_impact'] / divisor for y in years]

                    fig.add_trace(go.Scatter(
                        x=years, y=values, mode='lines+markers',
                        name='1610 & 4506', line=dict(width=3), marker=dict(size=8)
                    ))
            else:
                # Extract from stored results
                years, values = self._get_scenario_data(scenario, effect_type)
                if years:
                    fig.add_trace(go.Scatter(
                        x=years, y=values, mode='lines+markers',
                        name=f'Sector {scenario}', line=dict(width=3), marker=dict(size=8)
                    ))

        fig.update_layout(
            title=f'{effect_info[effect_type]["label"]} - IO Table Yearly Trends',
            xaxis_title='Year',
            yaxis_title=f'{effect_info[effect_type]["label"]} ({effect_info[effect_type]["unit"]})',
            height=600, hovermode='x unified', template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        if show_fig:
            fig.show()
        return fig

    def create_hydrogen_yearly_trends(self, effect_type: str, scenarios: List[str] = None,
                                      show_fig: bool = True):
        """
        Create yearly trends for hydrogen scenarios.
        Uses scenario_analyzer.integrate_hydrogen_H2S_H2T() for integrated data.

        Args:
            effect_type: 'productioncoeff', 'valueaddedcoeff', 'jobcoeff', 'directemploycoeff'
            scenarios: List of scenarios (default: ['H2S', 'H2T', 'H2S&H2T'])
            show_fig: Whether to display the figure

        Returns:
            Plotly Figure
        """
        if scenarios is None:
            scenarios = ['H2S', 'H2T', 'H2S&H2T']

        effect_info = {
            'productioncoeff': {'label': 'Indirect Production', 'unit': 'Billion Won'},
            'valueaddedcoeff': {'label': 'Value Added', 'unit': 'Billion Won'},
            'jobcoeff': {'label': 'Job Creation', 'unit': 'Billion Won'},
            'directemploycoeff': {'label': 'Direct Employment', 'unit': 'Person'}
        }

        fig = go.Figure()

        # Get integrated results if needed
        integrated_h2 = None
        if 'H2S&H2T' in scenarios:
            integrated_h2 = self.scenario_analyzer.integrate_hydrogen_H2S_H2T()

        for scenario in scenarios:
            if scenario == 'H2S&H2T' and integrated_h2:
                # Use integration method results
                if effect_type in integrated_h2:
                    years = sorted(integrated_h2[effect_type].keys())
                    values = [integrated_h2[effect_type][y]['total_impact'] /
                             (1000 if effect_type != 'directemploycoeff' else 1)
                             for y in years]

                    fig.add_trace(go.Scatter(
                        x=years, y=values, mode='lines+markers',
                        name='H2S & H2T', line=dict(width=3), marker=dict(size=8)
                    ))
            else:
                # Extract from stored results
                years, values = self._get_scenario_data(scenario, effect_type)
                if years:
                    fig.add_trace(go.Scatter(
                        x=years, y=values, mode='lines+markers',
                        name=scenario, line=dict(width=3), marker=dict(size=8)
                    ))

        fig.update_layout(
            title=f'{effect_info[effect_type]["label"]} - Hydrogen Scenarios Yearly Trends',
            xaxis_title='Year',
            yaxis_title=f'{effect_info[effect_type]["label"]} ({effect_info[effect_type]["unit"]})',
            height=600, hovermode='x unified', template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        if show_fig:
            fig.show()
        return fig

    def _get_scenario_data(self, sector: str, effect_type: str):
        """Extract data from scenario_analyzer.results."""
        # Find scenario index
        idx = None
        for i, row in self.scenario_analyzer.scenarios_data.iterrows():
            if str(row['sector']) == sector:
                idx = i
                break

        if idx is None or effect_type not in self.scenario_analyzer.results:
            return [], []

        scenario_key = f"scenario_{idx}"
        years = sorted(self.scenario_analyzer.results[effect_type].keys())
        values = []

        for year in years:
            if scenario_key in self.scenario_analyzer.results[effect_type][year]:
                total = self.scenario_analyzer.results[effect_type][year][scenario_key]['total_impact']
                # Convert units
                if effect_type in ['indirect_prod', 'indirect_import', 'value_added',
                                  'productioncoeff', 'valueaddedcoeff', 'jobcoeff']:
                    values.append(total / 1000)
                else:
                    values.append(total)
            else:
                values.append(0)

        return years, values

    def create_all_trends(self, output_dir: str = 'output/plotly_charts', save_html: bool = True):
        """
        Create all yearly trends visualizations (IO + Hydrogen).

        Args:
            output_dir: Directory to save HTML files
            save_html: Whether to save HTML files
        """
        import os
        if save_html:
            os.makedirs(f"{output_dir}/yearly_trends", exist_ok=True)
            os.makedirs(f"{output_dir}/yearly_trends_hydrogen", exist_ok=True)

        print("Creating all yearly trends visualizations...")
        print("=" * 80)

        # IO trends
        io_effects = ['indirect_prod', 'indirect_import', 'value_added', 'jobcoeff', 'directemploycoeff']
        io_labels = {
            'indirect_prod': 'indirect_production',
            'indirect_import': 'import',
            'value_added': 'value_added',
            'jobcoeff': 'job_creation',
            'directemploycoeff': 'direct_employment'
        }

        for effect in io_effects:
            print(f"\nCreating IO {io_labels[effect]} trends...")
            fig = self.create_io_yearly_trends(effect, show_fig=False)
            if save_html:
                filename = f"{output_dir}/yearly_trends/yearly_trends_{io_labels[effect]}.html"
                fig.write_html(filename)
                print(f"  ✅ Saved: {filename}")

        # Hydrogen trends
        h2_effects = ['productioncoeff', 'valueaddedcoeff', 'jobcoeff', 'directemploycoeff']
        h2_labels = {
            'productioncoeff': 'indirect_production',
            'valueaddedcoeff': 'value_added',
            'jobcoeff': 'job_creation',
            'directemploycoeff': 'direct_employment'
        }

        for effect in h2_effects:
            print(f"\nCreating Hydrogen {h2_labels[effect]} trends...")
            fig = self.create_hydrogen_yearly_trends(effect, show_fig=False)
            if save_html:
                filename = f"{output_dir}/yearly_trends_hydrogen/hydrogen_yearly_trends_{h2_labels[effect]}.html"
                fig.write_html(filename)
                print(f"  ✅ Saved: {filename}")

        print("\n" + "=" * 80)
        print("All yearly trends created!")
        print("=" * 80)

    def get_top_10_sectors(self, scenario: str, effect_type: str, year: int, n: int = 10):
        """
        Get top N sectors for a scenario using existing analyzer methods.

        Args:
            scenario: '1610', '4506', 'H2S', 'H2T', '1610&4506', 'H2S&H2T'
            effect_type: Effect type key
            year: Year to analyze
            n: Number of top sectors (default: 10)

        Returns:
            DataFrame with top N sectors or None
        """
        impacts_data = None

        # Get data from appropriate source
        if scenario == '1610&4506':
            # Use integration method
            integrated = self.scenario_analyzer.integrate_sectors_1610_4506()
            if integrated and effect_type in integrated and year in integrated[effect_type]:
                impacts_data = integrated[effect_type][year]['impacts']

        elif scenario == 'H2S&H2T':
            # Use integration method
            integrated = self.scenario_analyzer.integrate_hydrogen_H2S_H2T()
            if integrated and effect_type in integrated and year in integrated[effect_type]:
                impacts_data = integrated[effect_type][year]['impacts']

        else:
            # Single scenario - extract from results
            idx = None
            for i, row in self.scenario_analyzer.scenarios_data.iterrows():
                if str(row['sector']) == scenario:
                    idx = i
                    break

            if idx is not None and effect_type in self.scenario_analyzer.results:
                scenario_key = f"scenario_{idx}"
                if year in self.scenario_analyzer.results[effect_type]:
                    if scenario_key in self.scenario_analyzer.results[effect_type][year]:
                        impacts_data = self.scenario_analyzer.results[effect_type][year][scenario_key]['result']['impacts']

        # Process impacts data
        if not impacts_data:
            return None

        # Convert to DataFrame and get top N by absolute value
        import pandas as pd
        df = pd.DataFrame(impacts_data)
        df['abs_impact'] = df['impact'].abs()
        top_n = df.nlargest(n, 'abs_impact')[['sector_code', 'sector_name', 'impact']].reset_index(drop=True)

        return top_n

    def plot_top_10_sectors(self, scenario: str, effect_type: str, year: int, show_fig: bool = True):
        """
        Create bar chart for top 10 sectors.

        Args:
            scenario: '1610', '4506', 'H2S', 'H2T', '1610&4506', 'H2S&H2T'
            effect_type: Effect type key
            year: Year to analyze
            show_fig: Whether to display the figure

        Returns:
            Plotly Figure or None
        """
        top_10 = self.get_top_10_sectors(scenario, effect_type, year)

        if top_10 is None or len(top_10) == 0:
            print(f"No data available for {scenario} - {effect_type} - {year}")
            return None

        # Effect type labels
        effect_labels = {
            'indirect_prod': 'Indirect Production',
            'indirect_import': 'Import',
            'value_added': 'Value Added',
            'jobcoeff': 'Job Creation',
            'directemploycoeff': 'Direct Employment',
            'productioncoeff': 'Indirect Production',
            'valueaddedcoeff': 'Value Added'
        }

        # Determine unit
        if effect_type in ['jobcoeff']:
            if scenario in ['H2S', 'H2T', 'H2S&H2T']:
                unit = 'Billion Won'
            else:
                unit = 'Person'
        else:
            unit = 'Million Won'

        # Create colors based on positive/negative
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_10['impact']]

        # Create figure
        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=top_10['sector_name'],
            x=top_10['impact'],
            orientation='h',
            marker=dict(color=colors),
            text=top_10['impact'].apply(lambda x: f'{x:,.0f}'),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Code: ' + top_10['sector_code'].astype(str) +
                         '<br>Impact: %{x:,.0f}<extra></extra>'
        ))

        fig.update_layout(
            title=f'Top 10 Sectors - {scenario} | {effect_labels.get(effect_type, effect_type)} ({year})',
            xaxis_title=f'Impact ({unit})',
            yaxis_title='',
            height=600,
            showlegend=False,
            template='plotly_white'
        )

        if show_fig:
            fig.show()

        return fig

    def create_all_top10_charts(self, year: int = 2050, output_dir: str = 'output/plotly_charts',
                                save_html: bool = True):
        """
        Create all top 10 sector charts for a given year.

        Args:
            year: Year to analyze (default: 2050)
            output_dir: Directory to save HTML files
            save_html: Whether to save HTML files
        """
        import os
        if save_html:
            os.makedirs(output_dir, exist_ok=True)

        print(f"Creating top 10 sector charts for year {year}...")
        print("=" * 80)

        # Define all combinations
        combinations = [
            # IO Table scenarios
            ('1610', 'indirect_prod'),
            ('1610', 'indirect_import'),
            ('1610', 'jobcoeff'),
            ('4506', 'indirect_prod'),
            ('4506', 'indirect_import'),
            ('4506', 'jobcoeff'),
            ('1610&4506', 'indirect_prod'),
            ('1610&4506', 'indirect_import'),
            ('1610&4506', 'jobcoeff'),

            # Hydrogen scenarios
            ('H2S&H2T', 'productioncoeff'),
            ('H2S&H2T', 'jobcoeff'),
        ]

        for scenario, effect in combinations:
            print(f"\nCreating {scenario} - {effect}...")
            fig = self.plot_top_10_sectors(scenario, effect, year, show_fig=False)

            if fig and save_html:
                filename = f"{output_dir}/top10_{scenario}_{effect}_{year}.html"
                fig.write_html(filename)
                print(f"  ✅ Saved: {filename}")

        print("\n" + "=" * 80)
        print(f"All top 10 charts created for year {year}!")
        print("=" * 80)

    def create_code_h_heatmap(self, effect_type: str, year: int = 2030, top_n: int = 10,
                              output_path: str = None, show_fig: bool = True, use_plotly: bool = True):
        """
        Create a Code_H sector heatmap showing top N sectors for each Code_H category.
        
        Args:
            effect_type: Effect type (e.g., 'indirect_prod', 'jobcoeff')
            year: Year to analyze (default: 2030)
            top_n: Number of top sectors to show per Code_H (default: 10)
            output_path: Path to save the figure (default: auto-generated)
            show_fig: Whether to display the figure (default: True)
            use_plotly: Whether to use Plotly (True) or matplotlib (False) (default: True)
            
        Returns:
            Plotly Figure object or matplotlib Figure object
        """
        if output_path is None:
            ext = '.html' if use_plotly else '.png'
            output_path = f'code_h_heatmap_{effect_type}_{year}_top{top_n}{ext}'
        
        # Get aggregated results for the effect type and year
        if effect_type not in self.scenario_analyzer.aggregated_results:
            raise ValueError(f"Effect type '{effect_type}' not found in results")
        
        if year not in self.scenario_analyzer.aggregated_results[effect_type]:
            raise ValueError(f"Year {year} not found for effect type '{effect_type}'")
        
        # Prepare dataframe with sector impacts
        year_data = self.scenario_analyzer.aggregated_results[effect_type][year]
        sector_impacts = year_data['sector_impacts']
        
        # Build dataframe
        df_data = []
        for impact in sector_impacts:
            sector_code = impact['sector_code']
            sector_name = impact['sector_name']
            
            # Get Code_H and Product_H mapping from IO analyzer
            code_h = ''
            product_h = ''
            
            if hasattr(self.scenario_analyzer, 'io_analyzer') and self.scenario_analyzer.io_analyzer:
                code_h = self.scenario_analyzer.io_analyzer.basic_to_code_h.get(str(sector_code), '')
                if code_h:
                    product_h = self.scenario_analyzer.io_analyzer.code_h_to_product_h.get(code_h, code_h)
            
            # For H2 scenarios without Code_H mapping, use sector_code as Code_H
            if not code_h:
                code_h = str(sector_code)
                product_h = f"H2 Scenario ({sector_code})"
            
            # Include all sectors (with or without Code_H mapping)
            df_data.append({
                'Sector_Code': sector_code,
                'Sector_Name': sector_name,
                'Code_H': code_h,
                'Product_H': product_h,
                f'Year_{year}': impact['total_impact']
            })
        
        if not df_data:
            raise ValueError("No data available for this effect type and year")
        
        df = pd.DataFrame(df_data)
        
        # Call the appropriate heatmap function
        if use_plotly:
            fig = Visualization.plot_code_h_sector_top10_heatmap_plotly(
                df=df,
                effect=effect_type,
                year=year,
                top_n=top_n,
                output_path=output_path
            )
            if show_fig:
                fig.show()
        else:
            fig = Visualization.plot_code_h_sector_top10_heatmap(
                df=df,
                effect=None,  # Will use Year_{year} column
                year=year,
                top_n=top_n,
                output_path=output_path
            )
            if not show_fig:
                plt.close(fig)
        
        return fig

    @staticmethod
    def plot_code_h_sector_top10_heatmap(
        df,
        effect=None,
        year=2030,
        top_n=10,
        font_path='/System/Library/Fonts/Supplemental/AppleGothic.ttf',
        output_path='code_h_sector_top10_heatmap.png'
    ):
        """
        Plot a heatmap showing top N sectors within each Code_H group.
        X-axis: Product_H category names
        Y-axis: Rank (1 to top_n)
        Cell colors: Impact values (true values showing positive/negative)
        Cell labels: Sector names (formatted in three rows with smaller font to prevent overlap)
        Ranking: By absolute values (largest magnitude first)

        Args:
            df (pd.DataFrame): DataFrame with 'Sector_Name', 'Code_H', 'Product_H', and value columns.
            effect (str): Effect type to select column (e.g. 'indirect_prod').
            year (int): Year to select column (e.g. 2030).
            top_n (int): Number of top sectors to show for each Code_H (default: 10).
            font_path (str): Font file path for axis/labels.
            output_path (str): File path to save the generated heatmap PNG.
        """
        # Determine value column based on year and effect
        if effect is not None:
            possible_cols = [
                f"{effect}_{year}",
                f"{effect}",
                f"Year_{year}",
            ]
        else:
            possible_cols = [
                f"Year_{year}",
                'Year_2030'  # fallback
            ]

        # Find first match for value column
        value_column = None
        for col in possible_cols:
            if col in df.columns:
                value_column = col
                break
        if value_column is None:
            raise ValueError(f"Could not find value column matching {possible_cols} in dataframe columns: {df.columns}")

        # Prepare data - check for Product_H column
        required_cols = {'Sector_Name', 'Code_H', value_column}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing required columns in dataframe: wanted {required_cols}")

        # Use Product_H if available, otherwise fall back to Code_H
        use_product_h = 'Product_H' in df.columns
        category_col = 'Product_H' if use_product_h else 'Code_H'
        
        df_prep = df[['Sector_Name', 'Code_H', category_col, value_column]].copy()

        # Get all unique Code_H values (for grouping) and their Product_H names
        code_to_product = {}
        if use_product_h:
            for code_h in df_prep['Code_H'].unique():
                product_h = df_prep[df_prep['Code_H'] == code_h][category_col].iloc[0]
                code_to_product[code_h] = product_h
            all_codes = sorted(df_prep['Code_H'].unique())
        else:
            all_codes = sorted(df_prep['Code_H'].unique())
            code_to_product = {code: code for code in all_codes}

        # Create data structure for top N sectors per Code_H
        display_labels = [code_to_product[code] for code in all_codes]
        df_labels = pd.DataFrame(index=range(top_n), columns=display_labels)
        df_values = pd.DataFrame(index=range(top_n), columns=display_labels, dtype=float)

        for code in all_codes:
            df_group = df_prep[df_prep['Code_H'] == code].copy()
            
            # Sort by ABSOLUTE value (for ranking) but keep true values for coloring
            df_group['abs_value'] = df_group[value_column].abs()
            df_group_sorted = df_group.sort_values(by='abs_value', ascending=False).head(top_n)

            # Format sector names in THREE rows with very small font
            labels = []
            for name in df_group_sorted['Sector_Name'].tolist():
                # Split ALL names into exactly three lines
                words = name.split()
                if len(words) >= 3:
                    # Split into 3 lines - divide words into three parts
                    third = len(words) // 3
                    remainder = len(words) % 3
                    
                    # Distribute words evenly
                    if remainder == 0:
                        line1 = ' '.join(words[:third])
                        line2 = ' '.join(words[third:2*third])
                        line3 = ' '.join(words[2*third:])
                    elif remainder == 1:
                        line1 = ' '.join(words[:third+1])
                        line2 = ' '.join(words[third+1:2*third+1])
                        line3 = ' '.join(words[2*third+1:])
                    else:  # remainder == 2
                        line1 = ' '.join(words[:third+1])
                        line2 = ' '.join(words[third+1:2*third+2])
                        line3 = ' '.join(words[2*third+2:])
                    labels.append(f"{line1}\n{line2}\n{line3}")
                elif len(words) == 2:
                    # Two words: put on first two lines, empty third
                    labels.append(f"{words[0]}\n{words[1]}\n")
                else:
                    # Single word: put on first line, two empty lines
                    labels.append(f"{name}\n\n")
            
            # Use TRUE values for coloring (not absolute)
            values = df_group_sorted[value_column].tolist()

            # Pad if less than top_n sectors available
            padded_labels = labels + [''] * (top_n - len(labels))
            padded_values = values + [np.nan] * (top_n - len(values))

            display_label = code_to_product[code]
            df_labels[display_label] = padded_labels
            df_values[display_label] = padded_values

        # Create visualization
        font_properties = fm.FontProperties(fname=font_path)
        fig_height = max(12, top_n * 1.2)  # Adjusted height for three-line labels
        fig_width = max(16, len(all_codes) * 2.0)  # Increased width to prevent horizontal overlap

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Create heatmap with diverging colormap (to show positive/negative)
        sns.heatmap(
            df_values,
            annot=df_labels,
            fmt='s',
            cmap='RdBu_r',  # Red for positive, Blue for negative
            center=0,  # Center colormap at zero
            linewidths=0.8,  # Slightly thicker lines for better separation
            linecolor='lightgray',
            annot_kws={"size": 2.5, "color": "black", "fontproperties": font_properties, 
                       "va": "center", "ha": "center", "linespacing": 1.3},
            cbar_kws={'label': f'Impact Value'},
            ax=ax
        )

        # Configure axes
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='x', rotation=45, labelsize=10, pad=2)
        ax.tick_params(axis='y', labelsize=10)

        # Set title and labels
        ax.set_xlabel('Product Category', fontsize=11, fontproperties=font_properties)
        ax.set_ylabel('Rank (Top Sectors by Impact Magnitude)', fontsize=11, fontproperties=font_properties)

        # Set y-axis labels as ranks
        ax.set_yticklabels([f'#{i+1}' for i in range(top_n)])

        # Apply font properties to tick labels
        for label in ax.get_xticklabels():
            label.set_fontproperties(font_properties)
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_properties)

        # Configure colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_label(f'Impact Value (+ / -)', fontsize=10, fontproperties=font_properties)
        for t in cbar.ax.get_yticklabels():
            t.set_fontproperties(font_properties)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Heatmap saved to {output_path}")
        plt.show()

        return fig

    @staticmethod
    def plot_code_h_sector_top10_heatmap_plotly(
        df,
        effect=None,
        year=2030,
        top_n=10,
        output_path='code_h_sector_top10_heatmap.html'
    ):
        """
        Plot an interactive Plotly heatmap showing top N sectors within each Code_H group.
        X-axis: Product_H category names
        Y-axis: Rank (1 to top_n)
        Cell colors: Impact values (true values showing positive/negative)
        Cell labels: Sector names (formatted in multiple lines)
        Ranking: By absolute values (largest magnitude first)

        Args:
            df (pd.DataFrame): DataFrame with 'Sector_Name', 'Code_H', 'Product_H', and value columns.
            effect (str): Effect type to select column (e.g. 'indirect_prod').
            year (int): Year to select column (e.g. 2030).
            top_n (int): Number of top sectors to show for each Code_H (default: 10).
            output_path (str): File path to save the generated heatmap HTML.
            
        Returns:
            Plotly Figure object
        """
        # Determine value column based on year and effect
        if effect is not None:
            possible_cols = [
                f"{effect}_{year}",
                f"{effect}",
                f"Year_{year}",
            ]
        else:
            possible_cols = [
                f"Year_{year}",
                'Year_2030'  # fallback
            ]

        # Find first match for value column
        value_column = None
        for col in possible_cols:
            if col in df.columns:
                value_column = col
                break
        if value_column is None:
            raise ValueError(f"Could not find value column matching {possible_cols} in dataframe columns: {df.columns}")

        # Prepare data - check for Product_H column
        required_cols = {'Sector_Name', 'Code_H', value_column}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing required columns in dataframe: wanted {required_cols}")

        # Use Product_H if available, otherwise fall back to Code_H
        use_product_h = 'Product_H' in df.columns
        category_col = 'Product_H' if use_product_h else 'Code_H'
        
        df_prep = df[['Sector_Name', 'Code_H', category_col, value_column]].copy()

        # Get all unique Code_H values (for grouping) and their Product_H names
        code_to_product = {}
        if use_product_h:
            for code_h in df_prep['Code_H'].unique():
                product_h = df_prep[df_prep['Code_H'] == code_h][category_col].iloc[0]
                code_to_product[code_h] = product_h
            all_codes = sorted(df_prep['Code_H'].unique())
        else:
            all_codes = sorted(df_prep['Code_H'].unique())
            code_to_product = {code: code for code in all_codes}

        # Create data structure for top N sectors per Code_H
        display_labels = [code_to_product[code] for code in all_codes]
        z_values = []  # 2D array for heatmap colors
        text_labels = []  # 2D array for hover text
        hover_text = []  # 2D array for detailed hover info

        for rank in range(top_n):
            row_values = []
            row_labels = []
            row_hover = []
            
            for code in all_codes:
                df_group = df_prep[df_prep['Code_H'] == code].copy()
                
                # Sort by ABSOLUTE value (for ranking) but keep true values for coloring
                df_group['abs_value'] = df_group[value_column].abs()
                df_group_sorted = df_group.sort_values(by='abs_value', ascending=False)
                
                if rank < len(df_group_sorted):
                    row_data = df_group_sorted.iloc[rank]
                    sector_name = row_data['Sector_Name']
                    value = row_data[value_column]
                    
                    # Format sector name with line breaks - ALWAYS THREE LINES
                    words = sector_name.split()
                    if len(words) >= 3:
                        # Split into 3 lines - divide words into three parts
                        third = len(words) // 3
                        remainder = len(words) % 3
                        
                        # Distribute words evenly
                        if remainder == 0:
                            line1 = ' '.join(words[:third])
                            line2 = ' '.join(words[third:2*third])
                            line3 = ' '.join(words[2*third:])
                        elif remainder == 1:
                            line1 = ' '.join(words[:third+1])
                            line2 = ' '.join(words[third+1:2*third+1])
                            line3 = ' '.join(words[2*third+1:])
                        else:  # remainder == 2
                            line1 = ' '.join(words[:third+1])
                            line2 = ' '.join(words[third+1:2*third+2])
                            line3 = ' '.join(words[2*third+2:])
                        formatted_name = f"{line1}<br>{line2}<br>{line3}"
                    elif len(words) == 2:
                        # Two words: put on first two lines, empty third
                        formatted_name = f"{words[0]}<br>{words[1]}<br>"
                    else:
                        # Single word: just one line
                        formatted_name = sector_name
                    
                    row_values.append(value)
                    row_labels.append(formatted_name)
                    row_hover.append(
                        f"<b>{sector_name}</b><br>" +
                        f"Rank: #{rank + 1}<br>" +
                        f"Impact: {value:,.2f}<br>" +
                        f"Category: {code_to_product[code]}"
                    )
                else:
                    row_values.append(np.nan)
                    row_labels.append('')
                    row_hover.append('')
            
            z_values.append(row_values)
            text_labels.append(row_labels)
            hover_text.append(row_hover)

        # Create Plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=display_labels,
            y=[f'#{i+1}' for i in range(top_n)],
            text=text_labels,
            hovertext=hover_text,
            hoverinfo='text',
            texttemplate='%{text}',
            textfont={"size": 5},  # Very small font size for sector names
            colorscale='RdBu_r',  # Red for positive, Blue for negative
            zmid=0,  # Center colormap at zero
            colorbar=dict(
                title=dict(text="Impact Value", side="right"),
                tickformat=",",
            ),
            xgap=1,
            ygap=1,
        ))

        # Effect type labels for title
        effect_labels = {
            'indirect_prod': 'Indirect Production',
            'indirect_import': 'Indirect Import',
            'value_added': 'Value Added',
            'jobcoeff': 'Job Creation',
            'directemploycoeff': 'Direct Employment',
            'productioncoeff': 'Production Coeff',
            'valueaddedcoeff': 'Value Added Coeff'
        }
        
        # Scenario source information
        scenario_sources = {
            'indirect_prod': 'IO: 1610+4506',
            'indirect_import': 'IO: 1610+4506',
            'value_added': 'IO: 1610+4506',
            'jobcoeff': 'All: IO+H2',
            'directemploycoeff': 'All: IO+H2',
            'productioncoeff': 'H2: H2S+H2T',
            'valueaddedcoeff': 'H2: H2S+H2T'
        }
        
        title_text = effect_labels.get(effect, effect)
        source_text = scenario_sources.get(effect, 'All scenarios')
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Top {top_n} Sectors by Product Category<br><sub>{title_text} ({source_text}) | Year {year}</sub>',
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            xaxis=dict(
                title='Product Category',
                side='top',
                tickangle=-45,
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                title='Rank (Top Sectors by Impact Magnitude)',
                tickfont=dict(size=11),
                autorange='reversed'
            ),
            width=max(1000, len(all_codes) * 100),
            height=max(600, top_n * 60),
            template='plotly_white',
            margin=dict(l=150, r=150, t=150, b=100),
        )

        # Save to HTML
        fig.write_html(output_path)
        print(f"Interactive heatmap saved to {output_path}")
        
        return fig

    def plot_code_h_sector_ranking_heatmap(
        df,
        effect=None,
        year=2030,
        font_path='/System/Library/Fonts/Supplemental/AppleGothic.ttf',
        output_path='code_h_sector_ranking_heatmap.png'
    ):
        """
        Plot a heatmap showing sector rankings within each Code_H group, sorted by the specified value column.
        (Original version showing all sectors)

        Args:
            df (pd.DataFrame): DataFrame containing sector data.
            effect (str): Effect type to select column (e.g. 'indirect_prod').
            year (int): Year to select column (e.g. 2030).
            font_path (str): Font file path for axis/labels.
            output_path (str): File path to save the generated heatmap PNG.
        """
        # Determine value column based on year and effect (removed scenario logic)
        # Common formats: 'Year_2030', 'indirect_prod_2030', etc.
        if effect is not None:
            possible_cols = [
                f"{effect}_{year}",
                f"{effect}",
                f"Year_{year}",
            ]
        else:
            possible_cols = [
                f"Year_{year}",
                'Year_2030'  # fallback
            ]

        # Find first match for value column
        value_column = None
        for col in possible_cols:
            if col in df.columns:
                value_column = col
                break
        if value_column is None:
            raise ValueError(f"Could not find value column matching {possible_cols} in dataframe columns: {df.columns}")

        # --- 1. 데이터 준비 및 기준 값 설정 ---
        # 시각화에 필요한 컬럼만 추출
        if not {'Sector_Name', 'Code_H', value_column}.issubset(df.columns):
            raise ValueError(f"Missing required columns in dataframe: wanted {['Sector_Name', 'Code_H', value_column]}")

        df_prep = df[['Sector_Name', 'Code_H', value_column]].copy()

        # --- 2. X축(Code_H) 기준으로 데이터 재구성 ---
        all_codes = sorted(df_prep['Code_H'].unique())

        sorted_data = {}
        max_len = 0

        for code in all_codes:
            df_group = df_prep[df_prep['Code_H'] == code]
            df_group_sorted = df_group.sort_values(by=value_column, ascending=False)

            sorted_data[code] = {
                'labels': df_group_sorted['Sector_Name'].tolist(),
                'values': df_group_sorted[value_column].tolist()
            }
            if len(df_group_sorted) > max_len:
                max_len = len(df_group_sorted)

        # --- 3. 히트맵용 데이터프레임 생성 (Padding) ---
        df_labels = pd.DataFrame(index=range(max_len), columns=all_codes)
        df_values = pd.DataFrame(index=range(max_len), columns=all_codes, dtype=float)

        for code in all_codes:
            labels = sorted_data[code]['labels']
            values = sorted_data[code]['values']
            padded_labels = labels + [''] * (max_len - len(labels))
            padded_values = values + [np.nan] * (max_len - len(values))
            df_labels[code] = padded_labels
            df_values[code] = padded_values

        # --- 4. 시각화 ---
        font_properties = fm.FontProperties(fname=font_path)
        fig_height = max(60, max_len * 0.5)
        fig_width = max(20, len(all_codes) * 1.5)

        plt.figure(figsize=(fig_width, fig_height))

        ax = sns.heatmap(
            df_values,
            annot=df_labels,
            fmt='s',
            cmap='vlag',
            center=0,
            linewidths=0.5,
            linecolor='lightgray',
            annot_kws={"size": 8, "color": "black", "fontproperties": font_properties},
            cbar_kws={'label': f'Impact Value ({value_column})'}
        )

        # --- 5. 축 설정 ---
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='x', rotation=45)

        ax.set_title(
            f'Sector Ranking within each Code_H (Sorted by {value_column} Impact)',
            fontsize=20, pad=40, fontproperties=font_properties
        )
        ax.set_xlabel('Code_H', fontsize=14, fontproperties=font_properties)
        ax.set_ylabel(f'Rank (1 to {max_len})', fontsize=14, fontproperties=font_properties)

        for label in ax.get_xticklabels():
            label.set_fontproperties(font_properties)
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_properties)

        cbar = ax.collections[0].colorbar
        cbar.set_label(f'Impact Value ({value_column})', fontsize=13, fontproperties=font_properties)
        for t in cbar.ax.get_yticklabels():
            t.set_fontproperties(font_properties)

        ax.set_yticks([])

        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.show()


if __name__ == "__main__":
    from scenario_analyzer import ScenarioAnalyzer

    print("Loading and running scenario analysis...")
    analyzer = ScenarioAnalyzer()
    analyzer.run_all_scenarios()

    print("\nCreating visualizations...")
    viz = Visualization(analyzer)

    # Create all yearly trends
    #viz.create_all_trends(save_html=True)

    # Create all top 10 sector charts for year 2050
    #viz.create_all_top10_charts(year=2050, save_html=True)

    # Example: Create interactive Plotly Code_H heatmap for indirect production in 2030
    print("\nCreating interactive Code_H heatmap (Plotly)...")
    viz.create_code_h_heatmap(
        effect_type='indirect_prod',
        year=2030,
        top_n=10,
        use_plotly=True,
        output_path='code_h_heatmap_indirect_prod_2030.html'
    )
    
    # Example: Create static matplotlib Code_H heatmap for job creation in 2050
    viz.create_code_h_heatmap(
        effect_type='jobcoeff',
        year=2050,
        top_n=10,
        use_plotly=False,
        output_path='code_h_heatmap_jobcoeff_2050.png'
    )

    print("\nDone!")
