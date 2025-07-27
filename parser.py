import json
import re
from pathlib import Path
from datetime import datetime
import google.generativeai as genai
from typing import Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StartupAnalyzer:
    """Analyze startup conversation files and generate structured JSON output"""
    
    def __init__(self, gemini_api_key: str):
        """Initialize with Gemini API key"""
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.output_file = "startup_analyses.json"
        
    def analyze_text_file(self, file_path: str) -> Dict:
        """Analyze a single text file and return structured data"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = self._analyze_with_gemini(content)
            analysis['source_file'] = str(file_path)
            analysis['analysis_date'] = datetime.now().isoformat()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return self._create_fallback_analysis(file_path)

    def _analyze_with_gemini(self, text: str) -> Dict:
        """Use Gemini API to analyze the text and extract structured information"""
        prompt = f"""
        Analyze this startup conversation and extract key information in JSON format.
        
        Conversation:
        {text}
        
        Please analyze and return a JSON object with the following structure:
        {{
            "startup_summary": {{
                "name": "extracted or generated startup name",
                "idea": "concise startup idea description",
                "domain": "primary industry/domain",
                "target_audience": ["list of target audiences"],
                "unique_value_proposition": "what makes this startup unique",
                "geographic_focus": "primary market geography",
                "target_market_size": "local/regional/national/global"
            }},
            "product_details": {{
                "core_features": ["list of main product features"],
                "technology_stack": ["list of technologies used"],
                "platforms": ["list of supported platforms"],
                "development_stage": "idea/prototype/mvp/launched"
            }},
            "business_model": {{
                "revenue_streams": ["list of revenue sources"],
                "pricing_strategy": "description of pricing approach",
                "monetization_strategy": "how the product makes money",
                "projected_break_even": "estimated break-even timeline"
            }},
            "market_strategy": {{
                "key_differentiators": ["what makes it stand out"],
                "engagement_strategies": ["user retention approaches"],
                "partnership_opportunities": ["potential partnership types"]
            }},
            "conversation_insights": {{
                "key_discussions": ["main topics discussed"],
                "concerns_raised": ["challenges or concerns mentioned"],
                "next_steps": ["planned actions or next steps"]
            }},
            "tags": ["relevant keywords for categorization"]
        }}
        
        Rules:
        1. Be specific and accurate
        2. If information is not available, use "not_specified" or empty arrays
        3. Extract all possible details from the conversation
        4. Generate a meaningful startup name if not specified
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Clean and parse the JSON response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            return json.loads(response_text)
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._create_fallback_analysis(text)

    def _create_fallback_analysis(self, source: str) -> Dict:
        """Create a basic analysis structure when parsing fails"""
        return {
            "startup_summary": {
                "name": f"Startup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "idea": "not_specified",
                "domain": "not_specified",
                "target_audience": [],
                "unique_value_proposition": "not_specified",
                "geographic_focus": "not_specified",
                "target_market_size": "not_specified"
            },
            "product_details": {
                "core_features": [],
                "technology_stack": [],
                "platforms": [],
                "development_stage": "not_specified"
            },
            "business_model": {
                "revenue_streams": [],
                "pricing_strategy": "not_specified",
                "monetization_strategy": "not_specified",
                "projected_break_even": "not_specified"
            },
            "market_strategy": {
                "key_differentiators": [],
                "engagement_strategies": [],
                "partnership_opportunities": []
            },
            "conversation_insights": {
                "key_discussions": [],
                "concerns_raised": [],
                "next_steps": []
            },
            "tags": [],
            "source_file": str(source),
            "analysis_date": datetime.now().isoformat()
        }

    def process_folder(self, folder_path: str) -> List[Dict]:
        """Process all text files in a folder and return analyses"""
        analyses = []
        folder = Path(folder_path)
        
        if not folder.exists():
            logger.error(f"Folder not found: {folder_path}")
            return analyses
        
        for file_path in folder.glob("*.txt"):
            logger.info(f"Processing file: {file_path.name}")
            analysis = self.analyze_text_file(file_path)
            analyses.append(analysis)
            
        return analyses

    def save_to_json(self, analyses: List[Dict], append: bool = True):
        """Save analyses to JSON file, optionally appending to existing data"""
        existing_data = []
        
        if append and Path(self.output_file).exists():
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except Exception as e:
                logger.error(f"Error reading existing JSON file: {e}")
        
        combined_data = existing_data + analyses
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2)
        
        logger.info(f"Saved {len(analyses)} new analyses to {self.output_file} (total: {len(combined_data)})")

    def analyze_and_save(self, folder_path: str):
        """Complete pipeline: analyze folder and save results"""
        analyses = self.process_folder(folder_path)
        if analyses:
            self.save_to_json(analyses)
            print(f"Analysis complete. Results saved to {self.output_file}")
        else:
            print("No files were processed.")

# Example usage
if __name__ == "__main__":
    # Initialize with your Gemini API key
    analyzer = StartupAnalyzer(gemini_api_key="AIzaSyCJoMnwvwrnnIRBZ0noZIYxTMkGb_l6EC0")
    
    # Analyze a folder of text files and save results
    analyzer.analyze_and_save(r"D:\Desktop\New folder (4)\Conversations")