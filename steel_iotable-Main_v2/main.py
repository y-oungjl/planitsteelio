from libs.io_analyzer import IOTableAnalyzer
from libs.demandchange import DemandChangeAnalyzer

def main():
    """Main function to run the I-O analysis."""
    analyzer = IOTableAnalyzer()
    demandchange_analyzer = DemandChangeAnalyzer()
    
    print("\n" + "="*60)
    print("STEEL-COAL I-O TABLE ECONOMIC EFFECTS ANALYZER")
    print("="*60)
    
    while True:
        print("\nAvailable options:")
        print("1. List all sectors")
        print("2. Analyze economic effects")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            print("\nAll available sectors:")
            sectors = analyzer.get_sector_options()
            for display in sorted(sectors.values()):
                print(display)
        
        elif choice == '2':
            try:
                print("\nCoefficient types:")
                # print("A                 - Direct Total coefficients")
                # print("Am                - Direct Import coefficients") 
                # print("Ad                - Direct Domestic coefficients")
                print("indirect_prod     - Indirect Production (I-Ad)⁻¹")
                print("indirect_import   - Indirect Import coefficients")
                print("value_added       - Value-Added coefficients")
                print("jobcoeff          - Total Job Creation coefficients")
                print("directemploycoeff - Direct Employment coefficients")
                
                coeff_type = input("Select coefficient type: ").strip()
                valid_types = ['indirect_prod', 'indirect_import', 'value_added', 'jobcoeff', 'directemploycoeff']
                if coeff_type not in valid_types:
                    print("Invalid coefficient type. Using 'indirect_prod' as default.")
                    coeff_type = 'indirect_prod'
                
                #sector_input = input("Enter sector code (e.g., 111, 0111, or 2711): ").strip()

                print("Choose a scenario sector: ")
                print(demandchange_analyzer.demand_change_df['sector'])
                sector_input = str(input("Enter here: ").strip())
                demandchange_year = int(input("Enter scenario year: ").strip())
                
                # Convert input to proper format
                try:
                    sector_int = int(sector_input)
                    if sector_int < 1000:
                        sector_code = f"0{sector_int}"
                    else:
                        sector_code = str(sector_int)
                except ValueError:
                    # Already a string, use as is
                    sector_code = sector_input
                


                demand_change = demandchange_analyzer.get_scenario(sector_input, demandchange_year)

                print(demand_change)
               
                results = analyzer.calculate_direct_effects(sector_code, demand_change.item(), coeff_type)
                print("check1")
                analyzer.display_results(results)
                
            except ValueError as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")
        
        elif choice == '3':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()