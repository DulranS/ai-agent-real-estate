import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from typing import Dict, List, Optional
import logging
from collections import deque
import threading

# Try to import dotenv, but don't fail if it's not available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed. Please set GROQ_API_KEY environment variable manually.")

class RateLimiter:
    """
    Rate limiter for Groq API calls to stay within free tier limits
    """
    def __init__(self, requests_per_minute: int = 25, requests_per_day: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day
        self.minute_requests = deque()
        self.daily_requests = deque()
        self.lock = threading.Lock()
        
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        with self.lock:
            now = datetime.now()
            
            # Clean old requests from tracking
            minute_ago = now - timedelta(minutes=1)
            while self.minute_requests and self.minute_requests[0] < minute_ago:
                self.minute_requests.popleft()
                
            day_ago = now - timedelta(days=1)
            while self.daily_requests and self.daily_requests[0] < day_ago:
                self.daily_requests.popleft()
            
            # Check daily limit
            if len(self.daily_requests) >= self.requests_per_day:
                raise Exception(f"Daily API limit reached ({self.requests_per_day} requests). Please try again tomorrow.")
            
            # Check minute limit and wait if needed
            if len(self.minute_requests) >= self.requests_per_minute:
                # Need to wait until oldest request is > 1 minute old
                oldest_request = self.minute_requests[0]
                wait_until = oldest_request + timedelta(minutes=1, seconds=1)  # Add 1 sec buffer
                wait_time = (wait_until - now).total_seconds()
                
                if wait_time > 0:
                    print(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
            
            # Record this request
            self.minute_requests.append(now)
            self.daily_requests.append(now)
    
    def get_remaining_quota(self) -> Dict[str, int]:
        """Get remaining API quota"""
        with self.lock:
            now = datetime.now()
            
            # Clean old requests
            minute_ago = now - timedelta(minutes=1)
            while self.minute_requests and self.minute_requests[0] < minute_ago:
                self.minute_requests.popleft()
                
            day_ago = now - timedelta(days=1)
            while self.daily_requests and self.daily_requests[0] < day_ago:
                self.daily_requests.popleft()
            
            return {
                "minute_remaining": self.requests_per_minute - len(self.minute_requests),
                "daily_remaining": self.requests_per_day - len(self.daily_requests),
                "minute_used": len(self.minute_requests),
                "daily_used": len(self.daily_requests)
            }

class SriLankaTourismAgent:
    """
    AI Automation Agent for researching budget tourism hotspots in Sri Lanka
    Uses Groq AI to analyze tourism trends and generate investment reports
    Includes rate limiting to stay within Groq's free tier limits
    """
    
    def __init__(self, groq_api_key: Optional[str] = None, 
                 requests_per_minute: int = 25, 
                 requests_per_day: int = 100):
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(requests_per_minute, requests_per_day)
        
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
        Query Groq AI with a given prompt, respecting rate limits
        """
        try:
            # Check and wait for rate limits
            self.rate_limiter.wait_if_needed()
            
            # Log quota status
            quota = self.rate_limiter.get_remaining_quota()
            self.logger.info(f"API Quota - Minute: {quota['minute_used']}/{quota['minute_used'] + quota['minute_remaining']}, "
                           f"Daily: {quota['daily_used']}/{quota['daily_used'] + quota['daily_remaining']}")
            
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
            elif response.status_code == 429:
                # Rate limited by server - wait longer and retry once
                self.logger.warning("Server rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                return self.query_groq(prompt, max_tokens)  # Retry once
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
    
    def run_comprehensive_analysis(self, custom_locations: Optional[List[str]] = None, 
                                 max_locations: int = 2) -> Dict:
        """
        Run comprehensive analysis covering all aspects
        Limited number of locations to stay within API limits
        """
        locations = custom_locations if custom_locations else self.known_hotspots[:max_locations]
        # Ensure we don't exceed reasonable limits for free tier
        locations = locations[:max_locations]
        
        self.logger.info(f"Starting comprehensive Sri Lanka budget tourism analysis for {len(locations)} locations...")
        
        # Check if we have enough quota for the analysis
        quota = self.rate_limiter.get_remaining_quota()
        estimated_requests = 3 + len(locations) + 2  # market trends + report + locations + comparison + competition
        
        if quota['daily_remaining'] < estimated_requests:
            raise Exception(f"Insufficient daily API quota. Need ~{estimated_requests} requests, have {quota['daily_remaining']} remaining.")
        
        results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "locations_analyzed": locations,
            "reports": {},
            "api_usage": {
                "estimated_requests": estimated_requests,
                "starting_quota": quota
            }
        }
        
        try:
            # 1. General market trends
            self.logger.info("Researching budget tourism trends...")
            results["reports"]["market_trends"] = self.research_budget_tourism_trends()
            
            # 2. Comprehensive market report
            self.logger.info("Generating market report...")
            results["reports"]["market_report"] = self.generate_market_report()
            
            # 3. Location analysis (limited to save API calls)
            self.logger.info(f"Analyzing {len(locations)} locations...")
            results["reports"]["location_analysis"] = {}
            for i, location in enumerate(locations):
                self.logger.info(f"Analyzing {location} ({i+1}/{len(locations)})...")
                results["reports"]["location_analysis"][location] = self.analyze_location_potential(location)
            
            # 4. Location comparison
            self.logger.info("Comparing locations...")
            results["reports"]["location_comparison"] = self.compare_locations(locations)
            
            # 5. Competition analysis for top location only (to save API calls)
            if locations:
                top_location = locations[0]
                self.logger.info(f"Researching competition in {top_location}...")
                results["reports"]["competition_analysis"] = self.research_competition(top_location)
            
            # Final quota check
            final_quota = self.rate_limiter.get_remaining_quota()
            results["api_usage"]["ending_quota"] = final_quota
            results["api_usage"]["actual_requests_used"] = quota['daily_used'] - final_quota['daily_used']
            
            self.logger.info(f"Analysis complete. Used {results['api_usage']['actual_requests_used']} API requests.")
            
        except Exception as e:
            self.logger.error(f"Error during comprehensive analysis: {str(e)}")
            results["error"] = str(e)
            results["api_usage"]["ending_quota"] = self.rate_limiter.get_remaining_quota()
        
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
        
    def get_api_usage_summary(self) -> Dict:
        """Get current API usage summary"""
        quota = self.rate_limiter.get_remaining_quota()
        return {
            "quota_status": quota,
            "rate_limits": {
                "requests_per_minute": self.rate_limiter.requests_per_minute,
                "requests_per_day": self.rate_limiter.requests_per_day
            },
            "recommendations": self._get_usage_recommendations(quota)
        }
    
    def _get_usage_recommendations(self, quota: Dict) -> List[str]:
        """Get recommendations based on current usage"""
        recommendations = []
        
        if quota['daily_remaining'] < 10:
            recommendations.append("⚠️  Low daily quota remaining. Consider reducing analysis scope.")
        elif quota['daily_remaining'] < 20:
            recommendations.append("⚠️  Moderate daily quota remaining. Use wisely.")
        
        if quota['minute_remaining'] < 5:
            recommendations.append("⚠️  Near minute rate limit. Operations will be slower.")
        
        if quota['daily_remaining'] > 50:
            recommendations.append("✅ Good quota availability for comprehensive analysis.")
        
        return recommendations

# Usage Example and Main Execution
def main():
    """
    Example usage of the Sri Lanka Tourism Agent with rate limiting
    """
    try:
        # Initialize the agent with conservative rate limits for free tier
        # You can adjust these based on your specific Groq plan
        agent = SriLankaTourismAgent(
            requests_per_minute=25,  # Conservative limit for free tier
            requests_per_day=100     # Adjust based on your daily quota
        )
        
        print("Sri Lanka Budget Tourism Research Agent")
        print("=====================================")
        
        # Check API quota before starting
        usage = agent.get_api_usage_summary()
        print(f"\n📊 API Quota Status:")
        print(f"  Daily: {usage['quota_status']['daily_used']}/{usage['quota_status']['daily_used'] + usage['quota_status']['daily_remaining']}")
        print(f"  Minute: {usage['quota_status']['minute_used']}/{usage['quota_status']['minute_used'] + usage['quota_status']['minute_remaining']}")
        
        for rec in usage['recommendations']:
            print(f"  {rec}")
        
        # Ask user to confirm if quota is low
        if usage['quota_status']['daily_remaining'] < 20:
            response = input(f"\n⚠️  You have {usage['quota_status']['daily_remaining']} daily requests remaining. Continue? (y/n): ")
            if response.lower() != 'y':
                print("Analysis cancelled to preserve API quota.")
                return None, None
        
        # Example 1: Quick location analysis
        print("\n1. Analyzing Ella as investment location...")
        ella_analysis = agent.analyze_location_potential("Ella")
        if not ella_analysis['data'].startswith('Error'):
            print(f"✅ Analysis completed: {len(ella_analysis['data'])} characters of insights")
        else:
            print(f"❌ Analysis failed: {ella_analysis['data']}")
            return None, None
        
        # Check quota before continuing
        usage = agent.get_api_usage_summary()
        if usage['quota_status']['daily_remaining'] < 5:
            print(f"\n⚠️  Low quota remaining ({usage['quota_status']['daily_remaining']}). Stopping analysis.")
            return ella_analysis, {}
        
        # Example 2: Compare top locations (limited to 2 to save quota)
        print("\n2. Comparing top 2 locations...")
        comparison = agent.compare_locations(["Ella", "Kandy"])
        if not comparison['data'].startswith('Error'):
            print(f"✅ Comparison completed: {len(comparison['data'])} characters of analysis")
        else:
            print(f"❌ Comparison failed: {comparison['data']}")
        
        # Check quota before expensive operations
        usage = agent.get_api_usage_summary()
        if usage['quota_status']['daily_remaining'] < 8:
            print(f"\n⚠️  Insufficient quota for full analysis ({usage['quota_status']['daily_remaining']} remaining). Skipping comprehensive analysis.")
            return ella_analysis, {"comparison": comparison}
        
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
        if not recommendation['data'].startswith('Error'):
            print(f"✅ Recommendation generated: {len(recommendation['data'])} characters")
        else:
            print(f"❌ Recommendation failed: {recommendation['data']}")
        
        # Example 4: Limited comprehensive analysis (only if sufficient quota)
        usage = agent.get_api_usage_summary()
        if usage['quota_status']['daily_remaining'] >= 10:
            print(f"\n4. Running comprehensive analysis (limited scope to preserve quota)...")
            # Limit to 2 locations to stay within quota
            full_results = agent.run_comprehensive_analysis(["Ella", "Kandy"], max_locations=2)
            
            if "error" not in full_results:
                print("✅ Comprehensive analysis completed!")
                
                # Save results in all formats
                print("\n5. Saving reports in multiple formats...")
                saved_files = agent.save_all_formats(full_results)
                
                print(f"\nAnalysis complete! Reports saved:")
                for format_type, filepath in saved_files.items():
                    print(f"  {format_type.upper()}: {filepath}")
                
                # Show final quota usage
                final_usage = agent.get_api_usage_summary()
                print(f"\n📊 Final API Usage:")
                print(f"  Daily used: {final_usage['quota_status']['daily_used']}")
                print(f"  Daily remaining: {final_usage['quota_status']['daily_remaining']}")
                
                return full_results, saved_files
            else:
                print(f"❌ Comprehensive analysis failed: {full_results.get('error', 'Unknown error')}")
        else:
            print(f"\n4. Skipping comprehensive analysis - insufficient quota ({usage['quota_status']['daily_remaining']} remaining)")
            print("   Full comprehensive analysis requires ~6-8 API requests")
        
        return {"individual_analyses": [ella_analysis, comparison, recommendation]}, {}
        
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
        if "Daily API limit reached" in str(e):
            print("💡 Try again tomorrow when your daily quota resets.")
        else:
            print("Please check your API key and internet connection.")
        return None, None

if __name__ == "__main__":
    # Required packages to install:
    # pip install requests pandas openpyxl python-dotenv
    
    # Setup instructions:
    # 1. Get Groq API key from console.groq.com (free tier available)
    # 2. Set environment variable GROQ_API_KEY or create .env file
    # 3. Run the script
    
    # Note: This script includes rate limiting for Groq's free tier:
    # - 25 requests per minute (conservative)
    # - 100 requests per day (adjust based on your plan)
    # - Automatic quota checking and warnings
    
    print("🚀 Starting Sri Lanka Tourism Analysis with Rate Limiting...")
    results = main()
    if results[0]:
        print("\n" + "="*50)
        print("✅ SUCCESS: All reports generated successfully!")
        print("Check the generated files for detailed analysis.")
        print("💡 Rate limiting kept you within Groq's free tier limits.")
    else:
        print("\n" + "="*50)
        print("❌ FAILED: Please check the configuration and try again.")
        print("💡 Check your API quota at console.groq.com")