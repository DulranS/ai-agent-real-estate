import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from typing import Dict, List, Optional
import logging

# Try to import dotenv, but don't fail if it's not available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed. Please set GROQ_API_KEY environment variable manually.")

class SriLankaTourismAgent:
    """
    AI Automation Agent for researching budget tourism hotspots in Sri Lanka
    Uses Groq AI to analyze tourism trends and generate investment reports
    """
    
    def __init__(self, groq_api_key: Optional[str] = None):
        # Get API key from parameter, environment, or prompt user
        if groq_api_key:
            self.groq_api_key = groq_api_key
        elif os.getenv("GROQ_API_KEY"):
            self.groq_api_key = os.getenv("GROQ_API_KEY")
        else:
            raise ValueError("Please provide Groq API key either as parameter or set GROQ_API_KEY environment variable")
            
        self.base_url = "https://api.groq.com/openai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Known budget tourism hotspots from research
        self.known_hotspots = [
            "Ella", "Kandy", "Sigiriya", "Arugam Bay", "Negombo", 
            "Galle", "Nuwara Eliya", "Anuradhapura", "Mirissa", 
            "Unawatuna", "Trincomalee", "Polonnaruwa"
        ]
    
    def query_groq(self, prompt: str, max_tokens: int = 2000) -> str:
        """
        Query Groq AI with a given prompt
        """
        try:
            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "model": "llama3-8b-8192",
                "stream": False,
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                self.logger.error(f"Groq API error: {response.status_code} - {response.text}")
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            self.logger.error(f"Error querying Groq: {str(e)}")
            return f"Error: {str(e)}"
    
    def research_budget_tourism_trends(self) -> Dict:
        """
        Research current budget tourism trends in Sri Lanka
        """
        prompt = """
        Analyze the current budget tourism trends in Sri Lanka for 2024-2025. Focus on:
        
        1. Top 10 most popular budget destinations for backpackers and budget travelers
        2. Peak tourism seasons and months
        3. Average budget traveler spending per day
        4. Most popular accommodation types (hostels, guesthouses, budget hotels)
        5. Transportation preferences of budget travelers
        6. Popular activities and attractions for budget tourists
        7. Emerging budget tourism hotspots
        8. Current challenges in budget accommodation sector
        
        Provide specific data, statistics, and insights. Format as structured data.
        """
        
        response = self.query_groq(prompt)
        return {
            "timestamp": datetime.now().isoformat(),
            "research_type": "budget_tourism_trends",
            "data": response
        }
    
    def analyze_location_potential(self, location: str) -> Dict:
        """
        Analyze the potential of a specific location for budget accommodation investment
        """
        prompt = f"""
        Analyze {location}, Sri Lanka as a potential location for budget accommodation investment. Provide:
        
        1. Current tourism volume and growth trends
        2. Existing budget accommodation supply and occupancy rates
        3. Average room rates for budget accommodations
        4. Seasonality patterns
        5. Key attractions and activities drawing budget tourists
        6. Transportation accessibility
        7. Local competition analysis
        8. Investment potential score (1-10)
        9. Estimated ROI timeline
        10. Key challenges and opportunities
        11. Recommended accommodation type (hostel, guesthouse, budget hotel)
        12. Ideal room count and pricing strategy
        
        Be specific with numbers, costs, and market data where available.
        """
        
        response = self.query_groq(prompt)
        return {
            "location": location,
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "location_potential",
            "data": response
        }
    
    def compare_locations(self, locations: List[str]) -> Dict:
        """
        Compare multiple locations for investment potential
        """
        location_str = ", ".join(locations)
        prompt = f"""
        Compare these Sri Lankan locations for budget accommodation investment: {location_str}
        
        Create a comprehensive comparison including:
        1. Investment ranking (1st, 2nd, 3rd, etc.)
        2. Investment potential score for each (1-10)
        3. Pros and cons for each location
        4. Initial investment required (estimated)
        5. Expected monthly revenue
        6. Break-even timeline
        7. Risk assessment for each
        8. Market saturation level
        9. Growth potential
        10. Recommended investment strategy for each
        
        Provide a clear recommendation for the best investment opportunity.
        """
        
        response = self.query_groq(prompt)
        return {
            "locations": locations,
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "location_comparison",
            "data": response
        }
    
    def generate_market_report(self) -> Dict:
        """
        Generate comprehensive market report for Sri Lanka budget tourism
        """
        prompt = """
        Generate a comprehensive market report for budget accommodation investment in Sri Lanka:
        
        EXECUTIVE SUMMARY:
        - Market size and growth rate
        - Key opportunities and threats
        - Investment recommendations
        
        MARKET ANALYSIS:
        - Tourist arrival statistics (2023-2025)
        - Budget traveler demographics
        - Spending patterns
        - Seasonal trends
        
        COMPETITIVE LANDSCAPE:
        - Major budget accommodation chains
        - Independent operators
        - Market share distribution
        - Pricing strategies
        
        INVESTMENT OPPORTUNITIES:
        - Underserved markets
        - Emerging destinations
        - Investment requirements
        - Expected returns
        
        RISK ANALYSIS:
        - Political/economic risks
        - Tourism seasonality
        - Competition risks
        - Operational challenges
        
        RECOMMENDATIONS:
        - Top 3 investment locations
        - Optimal accommodation types
        - Pricing strategies
        - Marketing approaches
        
        Include specific numbers, statistics, and actionable insights.
        """
        
        response = self.query_groq(prompt)
        return {
            "report_type": "comprehensive_market_report",
            "timestamp": datetime.now().isoformat(),
            "data": response
        }
    
    def research_competition(self, location: str) -> Dict:
        """
        Research competition in a specific location
        """
        prompt = f"""
        Analyze the budget accommodation competition in {location}, Sri Lanka:
        
        1. List of major budget accommodations (hostels, guesthouses, budget hotels)
        2. Their room counts and pricing
        3. Occupancy rates and booking patterns
        4. Customer review analysis
        5. Unique selling points of each competitor
        6. Market gaps and opportunities
        7. Pricing strategies
        8. Marketing channels used
        9. Service quality assessment
        10. Competitive advantages you could leverage
        
        Provide specific names, prices, and actionable competitive intelligence.
        """
        
        response = self.query_groq(prompt)
        return {
            "location": location,
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "competition_analysis",
            "data": response
        }
    
    def generate_investment_recommendation(self, budget: float, preferences: Dict) -> Dict:
        """
        Generate personalized investment recommendation based on budget and preferences
        """
        pref_str = json.dumps(preferences, indent=2)
        
        prompt = f"""
        Generate a personalized budget accommodation investment recommendation for Sri Lanka:
        
        INVESTMENT BUDGET: ${budget:,.2f}
        
        PREFERENCES:
        {pref_str}
        
        Provide:
        1. Recommended location(s) within budget
        2. Optimal accommodation type and size
        3. Expected initial investment breakdown
        4. Revenue projections (monthly/yearly)
        5. ROI timeline and percentage
        6. Risk mitigation strategies
        7. Step-by-step implementation plan
        8. Licensing and legal requirements
        9. Operational considerations
        10. Marketing and booking strategies
        11. Staffing requirements
        12. Maintenance and ongoing costs
        
        Make it actionable with specific next steps and timelines.
        """
        
        response = self.query_groq(prompt)
        return {
            "budget": budget,
            "preferences": preferences,
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "investment_recommendation",
            "data": response
        }
    
    def run_comprehensive_analysis(self, custom_locations: Optional[List[str]] = None) -> Dict:
        """
        Run comprehensive analysis covering all aspects
        """
        locations = custom_locations if custom_locations else self.known_hotspots[:6]
        
        self.logger.info("Starting comprehensive Sri Lanka budget tourism analysis...")
        
        results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "locations_analyzed": locations,
            "reports": {}
        }
        
        try:
            # 1. General market trends
            self.logger.info("Researching budget tourism trends...")
            results["reports"]["market_trends"] = self.research_budget_tourism_trends()
            time.sleep(2)  # Rate limiting
            
            # 2. Comprehensive market report
            self.logger.info("Generating market report...")
            results["reports"]["market_report"] = self.generate_market_report()
            time.sleep(2)
            
            # 3. Location analysis
            self.logger.info("Analyzing individual locations...")
            results["reports"]["location_analysis"] = {}
            for location in locations[:3]:  # Limit to top 3 to manage API calls
                self.logger.info(f"Analyzing {location}...")
                results["reports"]["location_analysis"][location] = self.analyze_location_potential(location)
                time.sleep(2)
            
            # 4. Location comparison
            self.logger.info("Comparing locations...")
            results["reports"]["location_comparison"] = self.compare_locations(locations[:3])
            time.sleep(2)
            
            # 5. Competition analysis for top location
            top_location = locations[0]
            self.logger.info(f"Researching competition in {top_location}...")
            results["reports"]["competition_analysis"] = self.research_competition(top_location)
            
        except Exception as e:
            self.logger.error(f"Error during comprehensive analysis: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def save_report_txt(self, results: Dict, filename: Optional[str] = None) -> Optional[str]:
        """
        Save analysis results to a formatted TXT file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sri_lanka_tourism_report_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("SRI LANKA BUDGET TOURISM INVESTMENT ANALYSIS REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Report Generated: {results.get('analysis_timestamp', 'N/A')}\n")
                f.write(f"Locations Analyzed: {', '.join(results.get('locations_analyzed', []))}\n")
                f.write(f"Total Reports: {len(results.get('reports', {}))}\n\n")
                
                # Write each report section
                reports = results.get("reports", {})
                
                for report_type, report_data in reports.items():
                    f.write("=" * 60 + "\n")
                    f.write(f"{report_type.upper().replace('_', ' ')}\n")
                    f.write("=" * 60 + "\n")
                    
                    if isinstance(report_data, dict):
                        if "timestamp" in report_data:
                            f.write(f"Generated: {report_data['timestamp']}\n\n")
                        
                        if "data" in report_data:
                            f.write(str(report_data["data"]))
                            f.write("\n\n")
                        
                        # Handle location-specific data
                        if "location" in report_data:
                            f.write(f"Location: {report_data['location']}\n")
                        
                        if "locations" in report_data:
                            f.write(f"Locations Compared: {', '.join(report_data['locations'])}\n")
                        
                        if "budget" in report_data:
                            f.write(f"Budget Considered: ${report_data['budget']:,.2f}\n")
                    
                    f.write("\n" + "-" * 40 + "\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 80 + "\n")
            
            self.logger.info(f"TXT Report saved to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error creating TXT report: {str(e)}")
            return None
    
    def save_report_json(self, results: Dict, filename: Optional[str] = None) -> Optional[str]:
        """
        Save analysis results to JSON file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sri_lanka_tourism_report_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"JSON Report saved to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error creating JSON report: {str(e)}")
            return None
    
    def export_to_excel(self, results: Dict, filename: Optional[str] = None) -> Optional[str]:
        """
        Export key findings to Excel format
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sri_lanka_tourism_analysis_{timestamp}.xlsx"
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Create summary sheet
                summary_data = {
                    "Analysis Date": [results.get("analysis_timestamp", "")],
                    "Locations Analyzed": [", ".join(results.get("locations_analyzed", []))],
                    "Total Reports Generated": [len(results.get("reports", {}))],
                    "Report Types": [", ".join(results.get("reports", {}).keys())]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
                
                # Create detailed sheets for each report type
                for report_type, report_data in results.get("reports", {}).items():
                    if isinstance(report_data, dict) and "data" in report_data:
                        # Create a simple text export of the report
                        df_data = {
                            "Report Type": [report_type],
                            "Generated": [report_data.get("timestamp", "")],
                            "Content": [str(report_data.get("data", ""))[:32000]]  # Excel cell limit
                        }
                        # Ensure sheet name is valid (max 31 chars, no invalid chars)
                        sheet_name = report_type.replace('_', ' ')[:31]
                        pd.DataFrame(df_data).to_excel(writer, sheet_name=sheet_name, index=False)
                
            self.logger.info(f"Excel report saved to {filename}")
            return filename
                
        except Exception as e:
            self.logger.error(f"Error creating Excel report: {str(e)}")
            return None
    
    def save_all_formats(self, results: Dict, base_filename: Optional[str] = None) -> Dict[str, str]:
        """
        Save analysis results in all formats (TXT, JSON, Excel)
        """
        if not base_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"sri_lanka_tourism_report_{timestamp}"
        
        saved_files = {}
        
        # Save TXT format
        txt_file = self.save_report_txt(results, f"{base_filename}.txt")
        if txt_file:
            saved_files["txt"] = txt_file
        
        # Save JSON format
        json_file = self.save_report_json(results, f"{base_filename}.json")
        if json_file:
            saved_files["json"] = json_file
        
        # Save Excel format
        excel_file = self.export_to_excel(results, f"{base_filename}.xlsx")
        if excel_file:
            saved_files["excel"] = excel_file
        
        return saved_files

# Usage Example and Main Execution
def main():
    """
    Example usage of the Sri Lanka Tourism Agent
    """
    try:
        # Initialize the agent (will try to get API key from environment)
        agent = SriLankaTourismAgent()
        
        print("Sri Lanka Budget Tourism Research Agent")
        print("=====================================")
        
        # Example 1: Quick location analysis
        print("\n1. Analyzing Ella as investment location...")
        ella_analysis = agent.analyze_location_potential("Ella")
        print(f"Analysis completed: {len(ella_analysis['data'])} characters of insights")
        
        # Example 2: Compare top locations
        print("\n2. Comparing top 3 locations...")
        comparison = agent.compare_locations(["Ella", "Kandy", "Sigiriya"])
        print(f"Comparison completed: {len(comparison['data'])} characters of analysis")
        
        # Example 3: Generate investment recommendation
        print("\n3. Generating investment recommendation...")
        preferences = {
            "accommodation_type": "hostel",
            "target_guests": "backpackers",
            "location_preference": "mountain/nature",
            "room_count": "10-20",
            "amenities": ["wifi", "kitchen", "common_area"]
        }
        
        recommendation = agent.generate_investment_recommendation(50000, preferences)
        print(f"Recommendation generated: {len(recommendation['data'])} characters")
        
        # Example 4: Full comprehensive analysis
        print("\n4. Running comprehensive analysis...")
        full_results = agent.run_comprehensive_analysis(["Ella", "Kandy", "Arugam Bay"])
        
        # Save results in all formats
        print("\n5. Saving reports in multiple formats...")
        saved_files = agent.save_all_formats(full_results)
        
        print(f"\nAnalysis complete! Reports saved:")
        for format_type, filepath in saved_files.items():
            print(f"  {format_type.upper()}: {filepath}")
        
        return full_results, saved_files
        
    except ValueError as e:
        print(f"Configuration Error: {str(e)}")
        print("\nTo fix this:")
        print("1. Get a Groq API key from console.groq.com")
        print("2. Set environment variable: GROQ_API_KEY=your_api_key_here")
        print("3. Or create .env file with: GROQ_API_KEY=your_api_key_here")
        print("4. Or pass API key directly to SriLankaTourismAgent(api_key='your_key')")
        return None, None
    except Exception as e:
        print(f"Error running analysis: {str(e)}")
        print("Please check your API key and internet connection.")
        return None, None

if __name__ == "__main__":
    # Required packages to install:
    # pip install requests pandas openpyxl python-dotenv
    
    # Setup instructions:
    # 1. Get Groq API key from console.groq.com
    # 2. Set environment variable GROQ_API_KEY or create .env file
    # 3. Run the script
    
    results = main()
    if results[0]:
        print("\n" + "="*50)
        print("SUCCESS: All reports generated successfully!")
        print("Check the generated files for detailed analysis.")
    else:
        print("\n" + "="*50)
        print("FAILED: Please check the configuration and try again.")