# Steel Net Zero Transition Input-Output Analysis Tool

A comprehensive Input-Output (I-O) Table analysis tool for analyzing economic and employment impacts of hydrogen-reduced steel technologies for net zero transition. This system integrates conventional I-O table analysis with hydrogen scenario modeling to assess the transition from coal-based to hydrogen-based steel production.

## 🌟 Key Features

### **Multi-Table Analysis**
- **Conventional I-O Tables**: Korean I-O Table 2020 & 2023 analysis (380+ sectors)
- **Hydrogen Table Analysis**: Specialized hydrogen scenario modeling (H2S, H2T, etc.)
- **Integrated Analysis**: Combined assessment of conventional and hydrogen scenarios
- **Scenario Batch Processing**: Automated analysis across multiple years and scenarios

### **Comprehensive Impact Assessment**
- **Economic Effects**: 3 coefficient types (indirect production, import, value-added)
- **Employment Effects**: Job creation and direct employment across 165 sub-sectors
- **Hydrogen Effects**: Economic and employment effects for hydrogen scenarios
- **Multi-Year Analysis**: Time-series analysis from 2026 to 2050

### **Advanced Visualizations**
- **Interactive Plotly Charts**: Yearly trends, sector comparisons, heatmaps
- **Code_H Heatmaps**: Top sector impacts by product category with 3-line labels
- **Sector Ranking**: Top 10 sector analysis with impact magnitude visualization
- **Customizable Views**: Adjustable parameters for top N sectors, years, and effect types

### **Professional GUI Interface**
- **Streamlit Web Application**: User-friendly interface with tabbed navigation
- **Real-time Analysis**: Instant calculation and visualization updates
- **Scenario File Selection**: Choose between different scenario configurations
- **Export Capabilities**: Excel and CSV downloads with metadata

## 📦 Installation

### Requirements

```bash
pip install pandas streamlit openpyxl plotly seaborn matplotlib numpy
```

### Python Version
- Python 3.8 or higher recommended

## 🚀 Quick Start

### Launch Web Application
```bash
streamlit run main_gui.py
```

The application will open in your default web browser at `http://localhost:8501`

### Command Line Analysis
```bash
python main.py
```

## 📊 Application Structure

### Main Navigation

The application has three main modules accessible from the sidebar:

#### 1. **📋 Scenarios**
- View and select scenario files
- Preview scenario data
- Manage scenario configurations

#### 2. **📊 Result Tables** (5 Tabs)
- **🚀 Run Analysis**: Select scenario file and execute batch analysis
- **🔗 Integrated**: Summary tables combining all effect types (2026, 2030, 2040, 2050)
- **⚡ H2**: Hydrogen table analysis results
- **📊 Total**: Aggregated summary across all analyses
- **👤 Individual**: Detailed individual sector analysis

#### 3. **📈 Result Visualisation** (3 Tabs)
- **📈 Yearly Trends**: Time-series visualization of impacts
  - IO Table trends (1610=coal, 4506=renewables, 1610&4506=combined coal & renewables)
  - Hydrogen value chain trends (H2S=Hydrogen storage, H2T=Hydrogen transportation, H2S&H2T=both Hydrogen storage & transportation)
- **🗺️ Sector Maps**: Top 10 sector impact analysis
- **🔥 Code_H Heatmap**: Interactive heatmap by product category
  - Ranked by absolute values
  - Colored by true values (red=positive, blue=negative)
  - Top N sectors per category (configurable 5-20)
    
## 📖 User Guide

### Step 1: Load Scenario File

1. Navigate to **Tables** → **🚀 Run Analysis**
2. Select scenario file (e.g., `scenarios_1_2023.xlsx`)
3. Preview file contents (optional)
4. Click **"🚀 Run Complete Scenario Analysis"**
5. Wait for analysis to complete (~1-2 minutes)

**Check sidebar**: You should see ✅ with the loaded filename

### Step 2: View Integrated Results

1. Go to **Tables** → **🔗 Integrated** tab
2. Browse tabs for different effect types
3. View summary tables (2026, 2030, 2040, 2050)
4. Explore detailed sector impacts
5. Download data as needed

### Step 3: Generate Visualizations

#### Yearly Trends
1. Go to **Visualisation** → **📈 Yearly Trends**
2. Choose IO or Hydrogen table
3. Select effect type and scenarios
4. Click **"Generate"**

#### Code_H Heatmap
1. Go to **Visualisation** → **🔥 Code_H Heatmap**
2. Select:
   - Effect type (e.g., indirect_prod)
   - Year (2026, 2030, 2040, or 2050)
   - Top N sectors (5-20)
3. Click **"🎨 Generate Heatmap"**
4. View interactive heatmap with scenario information
5. Hover over cells for details

## 📁 Data Files

### Scenario Files (`data/`)
- **`scenarios_1_2020.xlsx`**: 2020 baseline scenarios
- **`scenarios_1_2023.xlsx`**: 2023 updated scenarios (recommended)

### Core Data Files
- **`iotable_2020.xlsx`**: Korean I-O Table 2020
- **`iotable_2023.xlsx`**: Korean I-O Table 2023 (latest)
- **`hydrogentable.xlsx`**: coefficients for Hydrogen value chain effects

### Scenario File Structure

Scenario Excel files contain:
- **Columns**: `input`, `sector`, and year columns (2026, 2027, ..., 2050)
- **input**: Data source ('iotable' or 'hydrogen')
- **sector**: Sector code ('1610', '4506', 'H2S', 'H2T', etc.)
- **Year columns**: Demand change values for each year

Example:
```
input      | sector | 2026    | 2027    | ... | 2050
-----------|--------|---------|---------|-----|----------
iotable    | 1610   | 1000000 | 1050000 | ... | 2000000
iotable    | 4506   | 500000  | 525000  | ... | 1000000
hydrogen   | H2S    | 100000  | 150000  | ... | 500000
hydrogen   | H2T    | 80000   | 120000  | ... | 400000
```

## 🔬 Analysis Types

### Economic Coefficients (I-O Table)

| Effect Type | Description | Unit | Scenarios |
|-------------|-------------|------|-----------|
| `indirect_prod` | Indirect Production (Leontief) | Million Won | 1610 + 4506 |
| `indirect_import` | Indirect Import | Million Won | 1610 + 4506 |
| `value_added` | Value Added (GDP) | Million Won | 1610 + 4506 |

### Hydrogen-specific Coefficients

| Effect Type | Description | Unit | Scenarios |
|-------------|-------------|------|-----------|
| `productioncoeff` | Production Inducing Effect | Million Won | H2S + H2T |
| `valueaddedcoeff` | Value Added Effect | Million Won | H2S + H2T |

### Employment Coefficients

| Effect Type | Description | Unit | Scenarios |
|-------------|-------------|------|-----------|
| `jobcoeff` | Total Job Creation | Persons | All (IO + H2) |
| `directemploycoeff` | Direct Employment | Persons | All (IO + H2) |

## 🎨 Visualization Features

### Code_H Heatmap

The Code_H heatmap provides a comprehensive view of sector impacts:

**Features**:
- **X-axis**: Product_H categories (Korean product names)
- **Y-axis**: Ranking (#1 to #10 or custom top N)
- **Cell Colors**: Impact values (diverging colormap)
  - 🔴 Red = Positive impact
  - 🔵 Blue = Negative impact  
  - ⚪ White = Near zero
- **Cell Text**: Sector names split into 3 lines (very small font)
- **Ranking Method**: By absolute values (magnitude)
- **Coloring Method**: By true values (shows direction)

**Interactive Features**:
- Hover for detailed information
- Zoom and pan capabilities
- Download as PNG or HTML
- Responsive layout

### Yearly Trends

Track how impacts evolve over time:
- Multiple scenarios on one chart
- Customizable effect types
- Separate IO and Hydrogen trend analysis
- Clear unit labeling (Billion Won vs Persons)

## 💾 Export Options

### Available Formats

1. **Excel (.xlsx)**
   - Multiple sheets per file
   - One sheet per effect type
   - Metadata included (analysis parameters, dates)
   - Formatted for easy reading

2. **CSV (.csv)**
   - UTF-8 with BOM encoding (Korean text support)
   - Fallback when Excel not available
   - Compatible with Excel and Google Sheets

3. **HTML (.html)**
   - Interactive Plotly charts
   - Fully functional offline
   - Shareable visualizations

### Export Contents

- **Summary Tables**: Aggregated impacts by year
- **Detailed Sector Data**: Individual sector impacts
- **Visualization Files**: Interactive charts and heatmaps
- **Complete Analysis**: All effect types in one file

## 📋 File Structure

```
steel_iotable/
├── main_gui.py                     # Main Streamlit application (1497 lines)
├── main.py                         # CLI interface (legacy)
│
├── libs/                           # Core library modules
│   ├── io_analyzer.py              # I-O Table analysis (583 lines)
│   ├── hydrogen_analyzer.py        # Hydrogen scenario analysis (242 lines)
│   ├── scenario_analyzer.py        # Batch scenario processor (995 lines)
│   ├── visualisation.py            # Visualization engine (1061 lines)
│   └── demandchange.py             # Demand change utilities
│
├── data/                           # Data files
│   ├── scenarios_1_2020.xlsx       # 2020 baseline scenarios
│   ├── scenarios_1_2023.xlsx       # 2023 updated scenarios ⭐
│   ├── iotable_2020.xlsx           # Korean I-O Table 2020
│   ├── iotable_2023.xlsx           # Korean I-O Table 2023 ⭐
│   └── hydrogentable.xlsx          # Hydrogen coefficients
│
├── output/                         # Analysis results
│   ├── combined_scenario_data.xlsx # back-up purpose
│   ├── total_impact_per_year.xlsx  # back-up purpose
│   └── scenario_analyzer produced/ # back-up purpose
│
├── archive/                        # Legacy code and documentation
└── README.md                       # This file
```

## 🎓 Analysis Methodology

### Economic Impact Formula

```
Impact = Coefficient Matrix × Demand Change Vector

For sector i:
Impact_i = Σ(C_ij × ΔD_j)

Where:
- C_ij: Coefficient from sector j to sector i
- ΔD_j: Demand change in sector j
```

### Employment Impact Formula

```
Jobs = Employment Coefficient Matrix × Demand Change

For sub-sector i:
Jobs_i = E_ij × ΔD_j

Where:
- E_ij: Job coefficient (jobs per billion won)
- ΔD_j: Demand change in basic sector j (mapped to sub-sector)
```

### Aggregation Methods

**Integrated Sectors (1610+4506)**:
```python
# Combine impacts from coal (1610) and renewable (4506) sectors
integrated_impact = impact_1610 + impact_4506
```

**Hydrogen Integration (H2S+H2T)**:
```python
# Combine hydrogen storage and transport scenarios
integrated_h2 = impact_H2S + impact_H2T
```
