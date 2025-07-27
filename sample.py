import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import google.generativeai as genai
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationAnalyzer:
    """Analyze startup conversations using Gemini API and prepare for RAG"""
    
    def __init__(self, gemini_api_key: str):
        """Initialize with Gemini API key"""
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
    def load_conversation_file(self, file_path: str) -> str:
        """Load conversation from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Loaded conversation from {file_path}")
            return content
        except Exception as e:
            logger.error(f"Error loading conversation file: {e}")
            raise
    
    def analyze_conversation_with_gemini(self, conversation_text: str) -> Dict:
        """Use Gemini API to analyze and summarize conversation"""
        
        analysis_prompt = f"""
        Analyze this startup conversation and extract key information in JSON format.
        
        Conversation:
        {conversation_text}
        
        Please analyze and return a JSON object with the following structure:
        {{
            "startup_summary": {{
                "name": "extracted or generated startup name",
                "idea": "concise startup idea description",
                "domain": "primary industry/domain (e.g., fitness, fintech, edtech, saas, healthcare, ecommerce)",
                "target_audience": ["list of target audiences"],
                "unique_value_proposition": "what makes this startup unique",
                "geographic_focus": "primary market geography (india, us, global, southeast_asia, europe, etc.)",
                "target_market_size": "local/regional/national/global"
            }},
            "business_analysis": {{
                "business_model": "freemium/subscription/marketplace/saas/transaction_based",
                "revenue_streams": ["list of revenue sources"],
                "monetization_details": "pricing and revenue information",
                "market_strategy": "go-to-market approach",
                "competitive_advantages": ["list of key advantages"],
                "market_type": "b2b/b2c/b2b2c"
            }},
            "technical_analysis": {{
                "technology_stack": ["web", "mobile", "ai", "blockchain", "wearables", etc.],
                "platforms": ["iOS", "Android", "Web", etc.],
                "key_features": ["list of main product features"],
                "development_stage": "idea/prototype/mvp/launched",
                "technical_complexity": "low/medium/high"
            }},
            "funding_analysis": {{
                "estimated_funding_stage": "pre_seed/seed/series_a/series_b",
                "funding_requirements": "estimated funding needs if mentioned",
                "timeline_to_market": "development timeline if mentioned",
                "key_metrics": ["user targets, revenue projections, etc."],
                "investor_preferences": "types of investors mentioned or preferred"
            }},
            "geographic_analysis": {{
                "team_location": "where is the founding team based",
                "primary_market": "main target market geography",
                "expansion_plans": "geographic expansion strategy if mentioned",
                "localization_needs": "does the product need local adaptation",
                "regulatory_considerations": "any regulatory or compliance factors"
            }},
            "vc_matching_tags": ["relevant tags for VC matching like ai, social, b2b, b2c, india, global, fintech, etc."],
            "conversation_insights": {{
                "key_discussions": ["main topics discussed"],
                "concerns_raised": ["challenges or concerns mentioned"],
                "next_steps": ["planned actions or next steps"],
                "team_collaboration": "insights about team dynamics",
                "investor_readiness": "how ready the team seems for fundraising"
            }}
        }}
        
        Make sure to be specific and accurate. If information is not available, use "not_specified" or empty arrays.
        Extract geographic context carefully as this is crucial for investor matching.
        """
        
        try:
            response = self.model.generate_content(analysis_prompt)
            response_text = response.text
            
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                analysis_result = json.loads(json_str)
            else:
                analysis_result = json.loads(response_text)
            
            logger.info("Successfully analyzed conversation with Gemini")
            return analysis_result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Raw response: {response.text}")
            return self._create_fallback_analysis(conversation_text)
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def _create_fallback_analysis(self, conversation_text: str) -> Dict:
        """Create basic analysis if Gemini parsing fails"""
        return {
            "startup_summary": {
                "name": f"Startup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "idea": conversation_text[:200] + "..." if len(conversation_text) > 200 else conversation_text,
                "domain": "technology",
                "target_audience": ["general"],
                "unique_value_proposition": "not_specified",
                "geographic_focus": "not_specified",
                "target_market_size": "not_specified"
            },
            "business_analysis": {
                "business_model": "not_specified",
                "revenue_streams": [],
                "monetization_details": "not_specified",
                "market_strategy": "not_specified",
                "competitive_advantages": [],
                "market_type": "not_specified"
            },
            "technical_analysis": {
                "technology_stack": [],
                "platforms": [],
                "key_features": [],
                "development_stage": "idea",
                "technical_complexity": "not_specified"
            },
            "funding_analysis": {
                "estimated_funding_stage": "pre_seed",
                "funding_requirements": "not_specified",
                "timeline_to_market": "not_specified",
                "key_metrics": [],
                "investor_preferences": "not_specified"
            },
            "geographic_analysis": {
                "team_location": "not_specified",
                "primary_market": "not_specified",
                "expansion_plans": "not_specified",
                "localization_needs": "not_specified",
                "regulatory_considerations": "not_specified"
            },
            "vc_matching_tags": ["technology", "startup"],
            "conversation_insights": {
                "key_discussions": [],
                "concerns_raised": [],
                "next_steps": [],
                "team_collaboration": "not_specified",
                "investor_readiness": "not_specified"
            }
        }
    
    def create_rag_document(self, analysis: Dict, source_file: str) -> Document:
        """Convert Gemini analysis into RAG-ready document with enhanced content"""
        
        startup = analysis["startup_summary"]
        business = analysis["business_analysis"]
        technical = analysis["technical_analysis"]
        funding = analysis["funding_analysis"]
        geographic = analysis["geographic_analysis"]
        insights = analysis["conversation_insights"]
        
        # Create comprehensive content for RAG with better structure
        content_parts = [
            "=== STARTUP PROFILE ===",
            f"Startup Name: {startup['name']}",
            f"Business Idea: {startup['idea']}",
            f"Industry Domain: {startup['domain'].title()}",
            f"Target Audience: {', '.join(startup['target_audience'])}",
            f"Unique Value Proposition: {startup['unique_value_proposition']}",
            f"Geographic Focus: {startup.get('geographic_focus', 'not_specified')}",
            f"Market Size: {startup.get('target_market_size', 'not_specified')}",
            "",
            "=== BUSINESS MODEL ===",
            f"Business Model Type: {business['business_model'].replace('_', ' ').title()}",
            f"Market Type: {business.get('market_type', 'not_specified').upper()}",
            f"Revenue Streams: {', '.join(business['revenue_streams'])}",
            f"Monetization Strategy: {business['monetization_details']}",
            f"Go-to-Market Strategy: {business['market_strategy']}",
            f"Competitive Advantages: {', '.join(business['competitive_advantages'])}",
            "",
            "=== TECHNICAL PROFILE ===",
            f"Technology Stack: {', '.join(technical['technology_stack'])}",
            f"Target Platforms: {', '.join(technical['platforms'])}",
            f"Key Product Features: {', '.join(technical['key_features'])}",
            f"Development Stage: {technical['development_stage'].replace('_', ' ').title()}",
            f"Technical Complexity: {technical.get('technical_complexity', 'not_specified').title()}",
            "",
            "=== FUNDING & INVESTMENT ===",
            f"Current Funding Stage: {funding['estimated_funding_stage'].replace('_', ' ').title()}",
            f"Funding Requirements: {funding['funding_requirements']}",
            f"Timeline to Market: {funding['timeline_to_market']}",
            f"Key Success Metrics: {', '.join(funding['key_metrics'])}",
            f"Investor Preferences: {funding.get('investor_preferences', 'not_specified')}",
            "",
            "=== GEOGRAPHIC CONTEXT ===",
            f"Team Location: {geographic.get('team_location', 'not_specified')}",
            f"Primary Target Market: {geographic.get('primary_market', 'not_specified')}",
            f"Expansion Plans: {geographic.get('expansion_plans', 'not_specified')}",
            f"Localization Needs: {geographic.get('localization_needs', 'not_specified')}",
            f"Regulatory Considerations: {geographic.get('regulatory_considerations', 'not_specified')}",
            "",
            "=== CONVERSATION INSIGHTS ===",
            f"Key Discussion Topics: {', '.join(insights['key_discussions'])}",
            f"Challenges & Concerns: {', '.join(insights['concerns_raised'])}",
            f"Planned Next Steps: {', '.join(insights['next_steps'])}",
            f"Team Dynamics: {insights['team_collaboration']}",
            f"Investment Readiness: {insights.get('investor_readiness', 'not_specified')}",
            "",
            f"=== MATCHING TAGS ===",
            f"VC Matching Keywords: {', '.join(analysis['vc_matching_tags'])}"
        ]
        
        content = "\n".join(content_parts)
        
        # Enhanced metadata for better retrieval
        metadata = {
            "startup_name": startup["name"],
            "domain": startup["domain"],
            "business_model": business["business_model"],
            "market_type": business.get("market_type", "not_specified"),
            "target_audience": startup["target_audience"],
            "technology_stack": technical["technology_stack"],
            "platforms": technical["platforms"],
            "funding_stage": funding["estimated_funding_stage"],
            "development_stage": technical["development_stage"],
            "geographic_focus": startup.get("geographic_focus", "not_specified"),
            "team_location": geographic.get("team_location", "not_specified"),
            "primary_market": geographic.get("primary_market", "not_specified"),
            "technical_complexity": technical.get("technical_complexity", "not_specified"),
            "tags": analysis["vc_matching_tags"],
            "source": source_file,
            "doc_type": "startup_conversation",
            "analyzed_date": datetime.now().isoformat(),
            "revenue_streams": business["revenue_streams"],
            "competitive_advantages": business["competitive_advantages"],
            "investor_readiness": insights.get("investor_readiness", "not_specified")
        }
        
        return Document(page_content=content, metadata=metadata)

class EnhancedVCRecommender:
    """Enhanced VC Recommender with improved RAG and geographic intelligence"""
    
    def __init__(self, gemini_api_key: str, embed_model="nomic-embed-text"):
        self.conversation_analyzer = ConversationAnalyzer(gemini_api_key)
        self.embed_model = embed_model
        self.startup_documents = []
        self.processed_conversations = []
        
    def load_vc_documents(self, folder: str) -> List[Document]:
        """Load VC documents with enhanced geographic and sector metadata"""
        files = [
            "companies.json", "funds.json", "deeptech_funds.json", "preseed_leads.json",
            "leadseed_funds.json", "prolific_preseed.json", "investors.json",
            "sheets_by_sector.json", "sheets_by_round.json", "sheets_by_geography.json", 
            "sheets_by_type.json", "vc_funds.json", "vc_individuals.json"
        ]
        
        docs = []
        for fname in files:
            try:
                fpath = Path(folder) / fname
                if not fpath.exists():
                    logger.warning(f"VC data file not found: {fpath}")
                    continue
                    
                with open(fpath, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                
                for entry in entries:
                    name = entry.get("name") or entry.get("sheet_name") or "Unnamed"
                    desc = entry.get("description", "")
                    
                    # Enhanced content with all available information
                    content_parts = [
                        f"VC Firm: {name}",
                        f"Description: {desc}",
                        f"Investment Focus: {entry.get('focus', 'General')}",
                        f"Geographic Focus: {entry.get('geography', entry.get('region', 'Global'))}",
                        f"Sector Specialization: {entry.get('sector', entry.get('region_or_sector', 'Multi-sector'))}",
                        f"Investment Stages: {', '.join(entry.get('stages', ['Pre-Seed', 'Seed']))}",
                        f"Website: {entry.get('website', 'Not available')}",
                        f"LinkedIn: {entry.get('linkedin', 'Not available')}",
                        f"Twitter: {entry.get('twitter', 'Not available')}",
                        f"Crunchbase: {entry.get('crunchbase', 'Not available')}",
                        f"Tags: {', '.join(entry.get('tags', [])) if isinstance(entry.get('tags', []), list) else entry.get('tags', '')}"
                    ]
                    content = "\n".join([p for p in content_parts if not p.endswith('Not available')])
                    
                    # Enhanced metadata with all contact info
                    metadata = {
                        "source": fname,
                        "website": entry.get("website", ""),
                        "linkedin": entry.get("linkedin", ""),
                        "twitter": entry.get("twitter", ""),
                        "crunchbase": entry.get("crunchbase", ""),
                        "focus": entry.get("focus", ""),
                        "sector": entry.get("sector", entry.get("region_or_sector", "")),
                        "geography": entry.get("geography", entry.get("region", "Global")).lower(),
                        "stages": entry.get("stages", ["Pre-Seed", "Seed", "Series A", "Series B+"]),
                        "tags": entry.get("tags", []),
                        "doc_type": "vc_profile",
                        "vc_name": name,
                        "is_indian_vc": self._is_indian_vc(entry),
                        "is_global_vc": self._is_global_vc(entry),
                        "is_us_vc": self._is_us_vc(entry),
                        "is_europe_vc": self._is_europe_vc(entry),
                        "is_asia_vc": self._is_asia_vc(entry)
                    }
                    docs.append(Document(page_content=content, metadata=metadata))
                    
            except Exception as e:
                logger.error(f"Failed loading {fname}: {e}")
                continue
        
        logger.info(f"Loaded {len(docs)} VC documents with enhanced contact info")
        return docs
    
    def _is_indian_vc(self, entry: Dict) -> bool:
        """Determine if VC has Indian focus"""
        indian_indicators = ['india', 'indian', 'mumbai', 'delhi', 'bangalore', 'chennai', 'hyderabad', 'pune']
        geography = entry.get("geography", "").lower()
        region = entry.get("region", "").lower()
        focus = entry.get("focus", "").lower()
        name = entry.get("name", "").lower()
        
        return any(indicator in text for text in [geography, region, focus, name] for indicator in indian_indicators)
    
    def _is_global_vc(self, entry: Dict) -> bool:
        """Determine if VC has global reach"""
        global_indicators = ['global', 'international', 'worldwide', 'multi-region']
        geography = entry.get("geography", "").lower()
        region = entry.get("region", "").lower()
        focus = entry.get("focus", "").lower()
        
        return any(indicator in text for text in [geography, region, focus] for indicator in global_indicators)
    
    def _is_us_vc(self, entry: Dict) -> bool:
        """Determine if VC has US focus"""
        us_indicators = ['usa', 'us', 'united states', 'silicon valley', 'san francisco', 'new york', 'boston']
        geography = entry.get("geography", "").lower()
        region = entry.get("region", "").lower()
        focus = entry.get("focus", "").lower()
        
        return any(indicator in text for text in [geography, region, focus] for indicator in us_indicators)
    
    def _is_europe_vc(self, entry: Dict) -> bool:
        """Determine if VC has European focus"""
        europe_indicators = ['europe', 'uk', 'united kingdom', 'germany', 'france', 'spain', 'italy']
        geography = entry.get("geography", "").lower()
        region = entry.get("region", "").lower()
        focus = entry.get("focus", "").lower()
        
        return any(indicator in text for text in [geography, region, focus] for indicator in europe_indicators)
    
    def _is_asia_vc(self, entry: Dict) -> bool:
        """Determine if VC has Asian focus (excluding India)"""
        asia_indicators = ['asia', 'singapore', 'hong kong', 'china', 'japan', 'south korea']
        geography = entry.get("geography", "").lower()
        region = entry.get("region", "").lower()
        focus = entry.get("focus", "").lower()
        
        return any(indicator in text for text in [geography, region, focus] for indicator in asia_indicators)

    def process_conversation_json(self, json_file: str) -> List[Document]:
        """Process JSON file containing startup conversations"""
        conversation_docs = []
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
                
            print("\n🔍 Processing Startup Conversations from JSON...")
            print("=" * 60)
            
            if not isinstance(conversations, list):
                conversations = [conversations]
                
            for conv in conversations:
                try:
                    # Convert conversation to string if it's a dict
                    if isinstance(conv, dict):
                        conversation_text = json.dumps(conv)
                    else:
                        conversation_text = str(conv)
                    
                    # Analyze with Gemini
                    analysis = self.conversation_analyzer.analyze_conversation_with_gemini(conversation_text)
                    
                    # Store analysis for acknowledgment
                    self.processed_conversations.append({
                        "file": json_file,
                        "startup_name": analysis["startup_summary"]["name"],
                        "domain": analysis["startup_summary"]["domain"],
                        "geographic_focus": analysis["startup_summary"].get("geographic_focus", "not_specified"),
                        "funding_stage": analysis["funding_analysis"]["estimated_funding_stage"],
                        "business_model": analysis["business_analysis"]["business_model"],
                        "market_type": analysis["business_analysis"].get("market_type", "not_specified"),
                        "technical_complexity": analysis["technical_analysis"].get("technical_complexity", "not_specified"),
                        "investor_readiness": analysis["conversation_insights"].get("investor_readiness", "not_specified"),
                        "analysis": analysis
                    })
                    
                    # Create RAG document
                    doc = self.conversation_analyzer.create_rag_document(analysis, json_file)
                    conversation_docs.append(doc)
                    
                    print(f"   ✅ Extracted: {analysis['startup_summary']['name']} ({analysis['startup_summary']['domain']})")
                    print(f"      🌍 Geographic Focus: {analysis['startup_summary'].get('geographic_focus', 'Global')}")
                    print(f"      💰 Funding Stage: {analysis['funding_analysis']['estimated_funding_stage'].replace('_', ' ').title()}")
                    print(f"      🏢 Business Model: {analysis['business_analysis']['business_model'].replace('_', ' ').title()}")
                    
                except Exception as e:
                    logger.error(f"Error processing conversation from JSON: {e}")
                    print(f"   ❌ Failed to process conversation: {e}")
                    continue
                    
            self.startup_documents = conversation_docs
            
            # Enhanced acknowledgment with detailed insights
            print(f"\n📊 JSON Analysis Complete!")
            print(f"   • Processed {len(conversation_docs)} startup conversations")
            print(f"   • Extracted insights from {len(self.processed_conversations)} startups")
            
            if self.processed_conversations:
                print(f"\n🎯 Startup Profiles Added to Knowledge Base:")
                
                # Group by domain for better visualization
                domain_groups = {}
                geographic_distribution = {}
                funding_stage_distribution = {}
                
                for conv in self.processed_conversations:
                    # Domain grouping
                    domain = conv['domain'].title()
                    if domain not in domain_groups:
                        domain_groups[domain] = []
                    domain_groups[domain].append(conv)
                    
                    # Geographic distribution
                    geo = conv['geographic_focus']
                    geographic_distribution[geo] = geographic_distribution.get(geo, 0) + 1
                    
                    # Funding stage distribution
                    stage = conv['funding_stage'].replace('_', ' ').title()
                    funding_stage_distribution[stage] = funding_stage_distribution.get(stage, 0) + 1
                
                # Display by domain
                for domain, startups in domain_groups.items():
                    print(f"\n   📂 {domain} ({len(startups)} startups):")
                    for startup in startups:
                        geo_info = f" ({startup['geographic_focus']})" if startup['geographic_focus'] != 'not_specified' else ""
                        print(f"      • {startup['startup_name']}{geo_info} - {startup['funding_stage'].replace('_', ' ').title()}")
                
                # Display insights summary
                print(f"\n📈 Knowledge Base Insights:")
                print(f"   🌍 Geographic Distribution: {dict(geographic_distribution)}")
                print(f"   💰 Funding Stages: {dict(funding_stage_distribution)}")
                print(f"   🏢 Domains Covered: {len(domain_groups)} ({', '.join(domain_groups.keys())}")
            
            print("=" * 60)
            
            return conversation_docs
            
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            print(f"❌ Failed to load JSON file: {e}")
            return []

    def setup_enhanced_recommender(self, vc_folder: str, conversation_json: str):
        """Setup recommender with enhanced retrieval using JSON conversation file"""
        
        # Load VC documents
        vc_docs = self.load_vc_documents(vc_folder)
        
        # Process conversation JSON file
        conversation_docs = self.process_conversation_json(conversation_json)
        
        # Combine all documents
        all_docs = vc_docs + conversation_docs
        
        if not all_docs:
            raise ValueError("No documents loaded. Check your folders.")
        
        logger.info(f"Total documents for RAG: {len(all_docs)} (VCs: {len(vc_docs)}, Startups: {len(conversation_docs)})")
        
        # Setup embeddings and vector store with increased retrieval
        embedder = OllamaEmbeddings(model=self.embed_model)
        clean_docs = filter_complex_metadata(all_docs)
        vectordb = Chroma.from_documents(clean_docs, embedder)
        
        # Significantly increase retrieval count for comprehensive matching
        retriever = vectordb.as_retriever(search_kwargs={"k": 50})  # Increased from 25 to 50
        
        # Setup LLM for comprehensive JSON output
        llm = ChatOllama(
            model="llama3.1:8b",
            num_ctx=8192,      # Increased context window significantly
            num_batch=1,
            temperature=0.05,
            format="json"      # Force JSON output
        )
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
    
    def _get_optimized_system_prompt(self, startup_idea: str) -> str:
        """Get the optimized system prompt with comprehensive geographic intelligence"""
        
        # Get knowledge base context
        kb_context = ""
        if self.processed_conversations:
            kb_context = f"""
KNOWLEDGE BASE CONTEXT:
You have access to {len(self.processed_conversations)} analyzed startup conversations with patterns including:
- Domains: {', '.join(set([conv['domain'] for conv in self.processed_conversations]))}
- Geographic focuses: {', '.join(set([conv['geographic_focus'] for conv in self.processed_conversations if conv['geographic_focus'] != 'not_specified']))}
- Funding stages: {', '.join(set([conv['funding_stage'] for conv in self.processed_conversations]))}

REFERENCE these patterns when making recommendations and ACKNOWLEDGE similar startups in your analysis.
"""
    
        return f"""
{kb_context}

COMPREHENSIVE VC RECOMMENDATION SYSTEM WITH GEOGRAPHIC INTELLIGENCE

STARTUP TO ANALYZE: "{startup_idea}"

CORE MISSION: Generate COMPREHENSIVE, geographically intelligent VC recommendations. Return ALL relevant VCs, not limited to a specific number. Be thorough and exhaustive.

=== ENHANCED GEOGRAPHIC INTELLIGENCE FRAMEWORK ===

1. **GEOGRAPHIC MATCHING PRIORITY**:
   - First match exact geographic focus (e.g., India, US, UK, Australia)
   - Then match regional focus (e.g., Asia, Europe, North America)
   - Finally consider global investors if no good local matches exist
   - For global startups, prioritize investors with relevant sector expertise regardless of location

2. **CONTACT INFORMATION REQUIREMENT**:
   - ALWAYS include ALL available contact information (website, LinkedIn, Twitter, Crunchbase)
   - Highlight investors with complete contact details
   - Never omit contact info if it exists in the knowledge base

3. **RECOMMENDATION QUANTITY**:
   - Return ALL relevant investors, not just a few
   - No artificial limit on number of recommendations
   - More recommendations are better as long as they're relevant

4. **GLOBAL STARTUP STRATEGY**:
   - For truly global startups, include:
     - 30% Top global VCs
     - 30% Regional specialists
     - 40% Local market experts in key regions
   - Ensure geographic diversity in recommendations

=== OUTPUT REQUIREMENTS ===

You MUST return a JSON object with the following structure:

{{
  "knowledge_base_insights": {{
    "similar_startups_referenced": ["list of similar startups from knowledge base"],
    "geographic_strategy": "detailed explanation of geographic matching approach"
  }},
  "recommended_vcs": [
    {{
      "name": "VC Name",
      "match_score": "1-100 score based on fit",
      "geographic_fit": "how their geographic focus aligns",
      "sector_alignment": "domain expertise match",
      "stage_compatibility": "investment stage alignment",
      "recommendation_reason": "specific rationale (min 2 sentences)",
      "contact_info": {{
        "website": "MUST include if available",
        "linkedin": "MUST include if available",
        "twitter": "MUST include if available",
        "crunchbase": "MUST include if available"
      }},
      "source_tags": ["list", "of", "relevant", "tags"],
      "priority_tier": "Tier 1 - Perfect Match/Tier 2 - Strong Fit/Tier 3 - Good Option"
    }}
  ],
  "total_vcs_analyzed": 0,
  "contact_info_availability": {{
    "with_website": 0,
    "with_linkedin": 0,
    "with_twitter": 0,
    "with_crunchbase": 0
  }}
}}
"""

    def print_enhanced_recommendations(self, result: Dict):
        """Print recommendations with enhanced formatting and contact information"""
        # Print knowledge base insights first
        kb_insights = result.get("knowledge_base_insights", {})
        if kb_insights:
            print("\n🧠 KNOWLEDGE BASE INTELLIGENCE APPLIED")
            print("=" * 60)
            
            if kb_insights.get("similar_startups_referenced"):
                print(f"\n📌 Similar Startups Referenced:")
                for startup in kb_insights["similar_startups_referenced"]:
                    print(f"   - {startup}")
            
            if kb_insights.get("geographic_strategy"):
                print(f"\n🌍 Geographic Matching Strategy:\n   {kb_insights['geographic_strategy']}")
        
        # Print VC recommendations
        vcs = result.get("recommended_vcs", [])
        total_analyzed = result.get("total_vcs_analyzed", len(vcs))
        contact_stats = result.get("contact_info_availability", {})
        
        print(f"\n📊 ANALYSIS RESULTS")
        print("=" * 60)
        print(f"\n🔍 Total VCs Analyzed: {total_analyzed}")
        print(f"\n🎯 Recommendations Generated: {len(vcs)}")
        print(f"\n📩 Contact Info Availability:")
        print(f"   • Websites: {contact_stats.get('with_website', 0)}")
        print(f"   • LinkedIn: {contact_stats.get('with_linkedin', 0)}")
        print(f"   • Twitter: {contact_stats.get('with_twitter', 0)}")
        print(f"   • Crunchbase: {contact_stats.get('with_crunchbase', 0)}")
        
        if not vcs:
            print("\n❌ No VC recommendations found.")
            return
        
        # Group by priority tier
        priority_groups = {
            "Tier 1 - Perfect Match": [],
            "Tier 2 - Strong Fit": [],
            "Tier 3 - Good Option": []
        }
        
        for vc in vcs:
            tier = vc.get("priority_tier", "Tier 3 - Good Option")
            priority_groups[tier].append(vc)
        
        # Print tier summaries
        print("\n🏆 PRIORITY BREAKDOWN")
        print("-" * 60)
        for tier, vc_list in priority_groups.items():
            if vc_list:
                print(f"   {tier.split('-')[0].strip()}: {len(vc_list)} VCs")
        
        # Print each tier's VCs
        for tier in priority_groups.keys():
            tier_vcs = priority_groups[tier]
            if not tier_vcs:
                continue
                
            print(f"\n🔹 {tier.upper()} ({len(tier_vcs)} VCs)")
            print("-" * 80)
            
            for i, vc in enumerate(tier_vcs, 1):
                print(f"\n{i}. {vc['name']}")
                print(f"   ⭐ Match Score: {vc.get('match_score', 'N/A')}")
                print(f"   📍 Geographic Fit: {vc.get('geographic_fit', 'Not specified')}")
                print(f"   🏢 Sector Alignment: {vc.get('sector_alignment', 'Not specified')}")
                print(f"   💡 Recommendation Reason: {vc.get('recommendation_reason', 'Not specified')}")
                
                # Print contact information - ensure we show all available
                contact_info = vc.get("contact_info", {})
                contacts = []
                if contact_info.get("website"):
                    contacts.append(f"🌐 {contact_info['website']}")
                if contact_info.get("linkedin"):
                    contacts.append(f"💼 {contact_info['linkedin']}")
                if contact_info.get("twitter"):
                    contacts.append(f"🐦 {contact_info['twitter']}")
                if contact_info.get("crunchbase"):
                    contacts.append(f"📊 {contact_info['crunchbase']}")
                
                if contacts:
                    print(f"   📩 Contact: {' | '.join(contacts)}")
                else:
                    print(f"   📩 Contact: No contact information available")
                
                # Print tags if available
                if vc.get("source_tags"):
                    tags = vc["source_tags"]
                    if isinstance(tags, list):
                        print(f"   🏷️ Tags: {', '.join(tags)}")
                    else:
                        print(f"   🏷️ Tags: {tags}")
                
                print("-" * 80)

    def get_enhanced_recommendations(self, startup_idea: str, vc_folder: str, conversation_json: str):
        """Get comprehensive VC recommendations with geographic intelligence"""
        
        # Setup recommender
        recommender = self.setup_enhanced_recommender(vc_folder, conversation_json)
        
        # Print knowledge base acknowledgment
        print(f"\n💡 Leveraging Enhanced Knowledge Base:")
        print("=" * 60)
        if self.processed_conversations:
            print(f"   • {len(self.processed_conversations)} startup conversation patterns analyzed")
            domain_count = {}
            geo_count = {}
            stage_count = {}
            
            for conv in self.processed_conversations:
                domain = conv['domain']
                domain_count[domain] = domain_count.get(domain, 0) + 1
                
                geo = conv['geographic_focus']
                geo_count[geo] = geo_count.get(geo, 0) + 1
                
                stage = conv['funding_stage']
                stage_count[stage] = stage_count.get(stage, 0) + 1
            
            print(f"   • Domain expertise: {', '.join([f'{d}({c})' for d, c in domain_count.items()])}")
            print(f"   • Geographic insights: {', '.join([f'{g}({c})' for g, c in geo_count.items() if g != 'not_specified'])}")
            print(f"   • Funding stage patterns: {', '.join([f'{s.replace('_', ' ')}({c})' for s, c in stage_count.items()])}")
        else:
            print("   • No conversation patterns available - using base VC database only")
        
        # Get OPTIMIZED SYSTEM PROMPT
        query = self._get_optimized_system_prompt(startup_idea)
        
        try:
            result = recommender.invoke(query)
            output = json.loads(result.get("result", "{}"))
            
            print(f"\n🎯 Comprehensive VC Analysis for: {startup_idea}")
            print("=" * 80)
            
            self.print_enhanced_recommendations(output)
            
            return output
            
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            print(f"❌ Error generating recommendations: {e}")
            return {}

if __name__ == "__main__":
    # Initialize with your Gemini API key
    recommender = EnhancedVCRecommender(gemini_api_key="AIzaSyCJoMnwvwrnnIRBZ0noZIYxTMkGb_l6EC0")
    
    # Get recommendations (replace paths with your actual data folders)
    recommendations = recommender.get_enhanced_recommendations(
        "AI-powered logistics platform across Global",
        vc_folder=r"D:\Desktop\New folder (4)",
        conversation_json=r"D:\Desktop\New folder (4)\startup_analyses.json"
    )