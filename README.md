# Steel Net Zero Transition Input-Output Analysis Tool

A comprehensive Input-Output (I-O) Table analysis tool for analyzing economic and employment impacts of hydrogen-reduced steel technologies for net zero transition. This system integrates conventional I-O table analysis with hydrogen scenario modeling to assess the transition from coal-based to hydrogen-based steel production.

## 🌟 Key Features

### **Multi-Table Analysis**
- **Conventional I-O Tables**: Korean I-O Table 2020 & 2023 analysis (380+ sectors)
- **Hydrogen Table Analysis**: Specialized hydrogen value chain sectors modeling (H2S, H2T, etc.)
- **Integrated Analysis**: Combined assessment of conventional and hydrogen value chain sectors
- **Scenario Batch Processing**: Automated analysis across multiple scenarios and years

### **Comprehensive Impact Assessment**
- **Economic Effects**: 3 coefficient types (indirect production, import, value-added)
- **Employment Effects**: Job creation and direct employment across 165 sub-sectors
- **Hydrogen Effects**: Economic and employment effects for hydrogen value chain sectors
- **Multi-Year Analysis**: Time-series analysis from 2026 to 2050

### **Advanced Visualizations**
- **Interactive Plotly Charts**: Yearly trends, sector comparisons, treemaps
- **Code_H Treemaps**: Impact visualization by product category with size and color coding
- **Sector Ranking**: Top 10 sector analysis with impact magnitude visualization
- **Customizable Views**: Adjustable parameters for years and effect types

### **Professional GUI Interface**
- **Streamlit Web Application**: User-friendly interface with tabbed navigation
- **Real-time Analysis**: Instant calculation and visualization updates
- **Scenario Data Management**: Automatic loading of Data_v11.xlsx scenario file
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

Note: The CLI interface requires the demandchange module which is currently under development.

## 📊 Application Structure

### Main Navigation

The application has three main modules accessible from the sidebar:

#### 1. **🚀 Run Analysis**
- Automatically loads Data_v11.xlsx scenario file
- Execute complete batch analysis across both scenarios (Scenario1=Optimized scenario and Scenario2=POSCO scenario)
- View analysis progress and completion status
- See loaded scenario sheets

#### 2. **📊 Table results** (5 Tabs)
- **🔀 Scenario Comparison**: Compare impacts across different scenarios
- **🔗 coal+renewable**: Combined analysis of sectors 1610 (Coal) and 4506 (Renewable)
- **⚡ H2 value chain**: Hydrogen value chain sector analysis (H2S=Hydrogen storage, H2T)
- **📊 coal+renewable+H2 value chain**: Integrated view of all sectors
- **👤 Individual**: Detailed individual sector analysis

#### 3. **📈 Visualisation** (4 Tabs)
- **📈 Yearly Trends**: Time-series visualization of impacts by sector
  - IO Table trends (1610=coal, 4506=renewables, 1610+4506=combined coal & renewables)
  - Hydrogen value chain trends (H2S=Hydrogen storage, H2T=Hydrogen transportation, H2S+H2T=both Hydrogen storage & transportation)
- **🗺️ Sector Maps**: Top 10 sector impact analysis
- **🌳 Code_H Treemap**: Interactive treemap by Code_H category
  - Four coefficient tabs: Indirect Production, Indirect Import, Value Added, Job Creation
  - Size represents impact magnitude (absolute values)
  - Color indicates direction: 🟦 Positive (blue/green) vs 🔴 Negative (red)
  - Shows coal+renewable+H2 value chain impacts
- **📊 Scenario comparison**: Compare both scenarios (Optimized vs POSCO) side-by-side

## 📖 User Guide

### Step 1: Run Scenario Analysis

1. Navigate to **Run Analysis** from the main menu
2. The system automatically uses `Data_v11.xlsx` from the data folder
3. Preview scenario data (optional)
4. Click **"🚀 Run Complete Scenario Analysis"**
5. Wait for analysis to complete (~1-2 minutes)

**Check sidebar**: You should see ✅ with "Data_v11.xlsx" displayed

### Step 2: View Analysis Results

1. Go to **Table results** from the main menu
2. Browse tabs for different analysis views:
   - **Scenario Comparison**: Compare different scenarios and effect types (you can find graphs, too)
   - **coal+renewable**: View integrated IO table results
   - **H2 value chain**: View hydrogen-specific impacts
   - **coal+renewable+H2 value chain**: See combined effects from all sectors
   - **Individual**: Explore detailed sector-by-sector data
3. View summary tables for target years (2026, 2030, 2040, 2050)
4. Explore detailed sector impacts
5. Download data as needed

### Step 3: Generate Visualizations

#### Yearly Trends
1. Go to **Visualisation** → **📈 Yearly Trends**
2. Choose IO or Hydrogen table
3. Select effect type and sectors
4. Click **"Generate"**

#### Code_H Treemap
1. Go to **Visualisation** → **🌳 Code_H Treemap**
2. Select:
   - Scenario sheet (Scenario1 or Scenario2)
   - Year (2026, 2030, 2040, or 2050)
3. Choose one of four coefficient tabs:
   - 💰 **Indirect Production**: Combined IO and H2 production effects
   - 🌐 **Indirect Import**: Import-inducing effects
   - 💎 **Value Added**: Combined IO and H2 value-added effects
   - 👥 **Job Creation**: Total employment effects
4. View interactive treemap automatically generated
5. Hover over rectangles for detailed values

## 📁 Data Files

### Scenario Files (`data/`)
- **`Data_v11.xlsx`**: Current data file containing two scenarios (Scenario1=Optimized and Scenario2=POSCO)

### Core Data Files
- **`iotable_2020.xlsx`**: Korean I-O Table 2020
- **`iotable_2023.xlsx`**: Korean I-O Table 2023 (latest)
- **`hydrogentable.xlsx`**: Coefficients for Hydrogen value chain effects

### Data File Structure

The Data_v11.xlsx file contains two main scenario sheets (Scenario1=Optimized and Scenario2=POSCO), plus supporting data sheets. Each scenario sheet has:
- **Columns**: `input`, `sector`, and year columns (2026, 2027, ..., 2050)
- **input**: Data source ('iotable_2023.xlsx' or 'hydrogentable.xlsx')
- **sector**: Sector code ('1610', '4506', 'H2S', 'H2T')
- **Year columns**: Demand change values for each year

Example:
```
input              | sector | 2026       | 2027       | ... | 2050
-------------------|--------|------------|------------|-----|------------
iotable_2023.xlsx  | 1610   | -660838    | -1206553   | ... | -9885162
iotable_2023.xlsx  | 4506   | 2529788    | 4029588    | ... | 80347640
hydrogentable.xlsx | H2S    | 0          | 0          | ... | 4028525
hydrogentable.xlsx | H2T    | 0          | 0          | ... | 4584183
```


## 🔬 Analysis Types

### Economic Coefficients (I-O Table)

| Effect Type | Description | Unit | Applicable Sectors |
|-------------|-------------|------|-----------|
| `indirect_prod` | Indirect Production (Leontief) | Million Won | 1610 + 4506 |
| `indirect_import` | Indirect Import | Million Won | 1610 + 4506 |
| `value_added` | Value Added (GDP) | Million Won | 1610 + 4506 |

### Hydrogen-specific Coefficients

| Effect Type | Description | Unit | Applicable Sectors |
|-------------|-------------|------|-----------|
| `productioncoeff` | Production Inducing Effect | Million Won | H2S + H2T |
| `valueaddedcoeff` | Value Added Effect | Million Won | H2S + H2T |

### Employment Coefficients

| Effect Type | Description | Unit | Applicable Sectors |
|-------------|-------------|------|-----------|
| `jobcoeff` | Total Job Creation | Persons | All (IO + H2) |
| `directemploycoeff` | Direct Employment | Persons | All (IO + H2) |

## 🎨 Visualization Features

### Code_H Treemap

The Code_H treemap provides a comprehensive view of sector impacts aggregated by Code_H categories:

**Features**:
- **Four Coefficient Tabs**:
  - 💰 Indirect Production (indirect_prod + productioncoeff)
  - 🌐 Indirect Import (indirect_import)
  - 💎 Value Added (value_added + valueaddedcoeff)
  - 👥 Job Creation (jobcoeff + directemploycoeff)
- **Rectangle Size**: Proportional to impact magnitude (absolute value)
  - Larger rectangle = Greater impact
- **Rectangle Color**: Indicates direction of impact
  - 🟦 Blue/Green = Positive impact (increases)
  - 🔴 Red/Orange = Negative impact (decreases)
- **Labels**: Display Product_H category name and value with +/- sign
- **Data Source**: coal+renewable+H2 value chain (1610+4506+H2S+H2T)

**Interactive Features**:
- Hover for detailed information (Code_H, Product_H, exact value)
- Automatic layout optimization
- Responsive design
- Download as HTML

### Yearly Trends

Track how impacts evolve over time:
- Multiple sectors on one chart
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
├── main_gui.py                     # Main Streamlit application (2599 lines)
├── main.py                         # CLI interface (85 lines, under development)
│
├── libs/                           # Core library modules
│   ├── __init__.py                 # Package initialization
│   ├── io_analyzer.py              # I-O Table analysis (583 lines)
│   ├── hydrogen_analyzer.py        # Hydrogen scenario analysis (242 lines)
│   ├── scenario_analyzer.py        # Batch scenario processor (1066 lines)
│   └── visualisation.py            # Visualization engine (1150 lines)
│
├── data/                           # Data files
│   ├── Data_v11.xlsx               # Current data file with multiple scenarios ⭐
│   ├── iotable_2020.xlsx           # Korean I-O Table 2020
│   ├── iotable_2023.xlsx           # Korean I-O Table 2023 ⭐
│   └── hydrogentable.xlsx          # Hydrogen coefficients
│
├── RAS trial/                      # RAS methodology experiments, back-up purpose
│   ├── rassourcecode.py            # RAS algorithm implementation, back-up purpose
│   ├── rassourcecode_gras.py       # GRAS algorithm implementation, back-up purpose
│   └── output/                     # RAS estimation outputs, back-up purpose
│
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
