import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import time
import re
from dataclasses import dataclass
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from urllib.parse import urljoin, urlparse

# Configure page
st.set_page_config(
    page_title="UNL Geoportal Intelligent Search",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GISDataItem:
    """Data structure for GIS datasets"""
    id: str
    title: str
    description: str
    tags: List[str]
    owner: str
    created: str
    modified: str
    type: str
    url: str
    portal_url: str
    thumbnail: Optional[str] = None
    extent: Optional[Dict] = None
    num_views: Optional[int] = None
    snippet: Optional[str] = None

class ArcGISPortalConnector:
    """Enhanced connector with multiple fallback methods"""
    
    def __init__(self, base_url: str = "https://geoportal.unl.edu"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 30
        
        # Add common headers to appear more like a browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/html, */*',
            'Accept-Language': 'en-US,en;q=0.9'
        })
        
        self.working_endpoint = None
        self.api_type = None
        self.find_working_endpoint()
    
    def test_endpoint(self, url: str, params: dict = None) -> tuple:
        """Test an endpoint and return success status and data"""
        try:
            if params is None:
                params = {'f': 'json'}
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    return True, data, response.status_code
                except json.JSONDecodeError:
                    return False, response.text, response.status_code
            else:
                return False, f"HTTP {response.status_code}", response.status_code
                
        except Exception as e:
            return False, str(e), None
    
    def find_working_endpoint(self):
        """Try multiple endpoint patterns to find working API"""
        
        # Different API endpoint patterns to try
        endpoint_patterns = [
            # Standard ArcGIS Portal patterns
            f"{self.base_url}/portal/sharing/rest/search",
            f"{self.base_url}/portal/sharing/rest/content/items", 
            f"{self.base_url}/sharing/rest/search",
            f"{self.base_url}/arcgis/sharing/rest/search",
            f"{self.base_url}/server/rest/services",
            f"{self.base_url}/rest/services",
            
            # Alternative patterns
            f"{self.base_url}/portal/rest/services",
            f"{self.base_url}/geoportal/rest/find/document",
            f"{self.base_url}/catalog/search/resource/details.page",
        ]
        
        # Test parameters for search endpoints
        test_params = [
            {'q': '*', 'f': 'json', 'num': 1},
            {'f': 'json', 'num': 1},
            {'q': 'census', 'f': 'json', 'num': 1},
            {'searchText': '*', 'f': 'json', 'max': 1},
        ]
        
        st.sidebar.write("**🔍 Testing API endpoints...**")
        
        for endpoint in endpoint_patterns:
            for params in test_params:
                success, data, status_code = self.test_endpoint(endpoint, params)
                
                if success and isinstance(data, dict):
                    # Check if this looks like a valid response
                    if ('results' in data or 'items' in data or 'services' in data or 
                        'records' in data or 'documents' in data):
                        
                        self.working_endpoint = endpoint
                        self.api_type = "search"
                        st.sidebar.success(f"✅ Found working endpoint!")
                        st.sidebar.write(f"URL: {endpoint}")
                        st.sidebar.write(f"Params: {params}")
                        
                        # Store the working params
                        self.working_params = params
                        return
        
        st.sidebar.error("❌ No working API endpoints found")
        st.sidebar.write("**Trying alternative approaches...**")
        
        # Try to get portal info
        info_endpoints = [
            f"{self.base_url}/portal/sharing/rest/portals/info",
            f"{self.base_url}/sharing/rest/portals/info",
            f"{self.base_url}/arcgis/sharing/rest/portals/info"
        ]
        
        for endpoint in info_endpoints:
            success, data, status_code = self.test_endpoint(endpoint)
            if success:
                st.sidebar.info(f"ℹ️ Portal info available at: {endpoint}")
                break
    
    def search_content(self, query: str = "*", num: int = 100, start: int = 1, group_id: str = None) -> Dict:
        """Search for content using the working endpoint with optional group filtering"""
        if not self.working_endpoint:
            return {'results': [], 'total': 0}
        
        # Use the working parameters as base
        params = self.working_params.copy()
        
        # For group filtering, we need to modify the approach
        if group_id:
            # Try different group filtering approaches
            group_queries = [
                f'({query}) AND group:{group_id}',
                f'({query}) AND orgid:{group_id}',
                f'{query} group:{group_id}',
                query  # Fallback to no group filter if others fail
            ]
            
            # Try each approach
            for group_query in group_queries:
                test_params = params.copy()
                
                if 'q' in test_params:
                    test_params['q'] = group_query
                elif 'searchText' in test_params:
                    test_params['searchText'] = group_query
                
                # Add additional group-specific parameters
                if group_id and group_query != query:  # Not the fallback
                    test_params['groupId'] = group_id
                
                # Test this approach
                try:
                    response = self.session.get(self.working_endpoint, params=test_params, timeout=15)
                    if response.status_code == 200:
                        result = response.json()
                        if self._has_results(result):
                            params = test_params  # Use this working approach
                            break
                except:
                    continue
        else:
            # No group filtering
            if 'q' in params:
                params['q'] = query
            elif 'searchText' in params:
                params['searchText'] = query
        
        # Update pagination
        if 'num' in params:
            params['num'] = min(num, 100)
            params['start'] = start
        elif 'max' in params:
            params['max'] = min(num, 100)
            params['start'] = start
        
        try:
            response = self.session.get(self.working_endpoint, params=params, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Normalize response format
                if 'results' in result:
                    return {
                        'results': result['results'],
                        'total': result.get('total', len(result['results']))
                    }
                elif 'items' in result:
                    return {
                        'results': result['items'],
                        'total': len(result['items'])
                    }
                elif 'records' in result:
                    return {
                        'results': result['records'],
                        'total': result.get('totalResults', len(result['records']))
                    }
                elif isinstance(result, list):
                    return {
                        'results': result,
                        'total': len(result)
                    }
                    
        except Exception as e:
            logger.error(f"Search failed: {e}")
        
        return {'results': [], 'total': 0}
    
    def _has_results(self, result):
        """Check if API response has results"""
        if isinstance(result, dict):
            return (bool(result.get('results')) or 
                   bool(result.get('items')) or 
                   bool(result.get('records')))
        elif isinstance(result, list):
            return len(result) > 0
        return False
        
        try:
            response = self.session.get(self.working_endpoint, params=params, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Normalize response format
                if 'results' in result:
                    return {
                        'results': result['results'],
                        'total': result.get('total', len(result['results']))
                    }
                elif 'items' in result:
                    return {
                        'results': result['items'],
                        'total': len(result['items'])
                    }
                elif 'records' in result:
                    return {
                        'results': result['records'],
                        'total': result.get('totalResults', len(result['records']))
                    }
                elif isinstance(result, list):
                    return {
                        'results': result,
                        'total': len(result)
                    }
                    
        except Exception as e:
            logger.error(f"Search failed: {e}")
        
        return {'results': [], 'total': 0}
    
    def get_all_content(self, max_items: int = 1000, group_id: str = None) -> List[Dict]:
        """Get all available content with progress tracking and optional group filtering"""
        all_items = []
        start = 1
        num_per_request = 50  # Smaller batches for better reliability
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Show group filter status
        if group_id:
            st.sidebar.info(f"🔒 Filtering by group ID: {group_id}")
        
        # Try different search strategies
        search_terms = ["*", "", "type:*", "census", "map", "data"]
        
        for search_term in search_terms:
            if all_items:  # If we got data, stop trying
                break
                
            status_text.text(f"Trying search term: '{search_term}'...")
            current_start = 1
            items_from_this_search = []
            
            while len(items_from_this_search) < max_items:
                result = self.search_content(
                    query=search_term, 
                    num=num_per_request, 
                    start=current_start,
                    group_id=group_id  # Pass group ID instead of name
                )
                
                if not result['results']:
                    break
                
                items_from_this_search.extend(result['results'])
                
                # Update progress
                progress = min(len(items_from_this_search) / max_items, 1.0)
                progress_bar.progress(progress)
                
                filter_text = f" (filtered by group {group_id})" if group_id else ""
                status_text.text(f"Found {len(items_from_this_search)} items with '{search_term}'{filter_text}")
                
                # Check if we got fewer results than requested (end of results)
                if len(result['results']) < num_per_request:
                    break
                    
                current_start += num_per_request
                time.sleep(0.2)  # Be nice to the server
            
            if items_from_this_search:
                all_items = items_from_this_search
                filter_text = f" using group filter '{group_id}'" if group_id else ""
                st.sidebar.success(f"✅ Loaded {len(all_items)} items using search term: '{search_term}'{filter_text}")
                break
        
        # If still no results, try sample data (but only if no group filter)
        if not all_items and not group_id:
            st.warning("⚠️ Could not load data from API. Using sample data for demonstration.")
            all_items = self.create_sample_data()
        elif not all_items and group_id:
            st.warning(f"⚠️ No data found for group '{group_id}'. Trying without group filter...")
            # Try once more without group filter as fallback
            fallback_result = self.search_content(query="*", num=50, start=1, group_id=None)
            if fallback_result['results']:
                st.info("ℹ️ Showing all available data (group filter may not be working)")
                all_items = fallback_result['results']
            else:
                st.error("❌ No data available from the API")
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Total items loaded: {len(all_items)}")
        
        return all_items[:max_items]
    
    def create_sample_data(self) -> List[Dict]:
        """Create sample data for demonstration when API fails"""
        sample_data = [
            {
                'id': 'sample_census_001',
                'title': 'Nebraska Census Population Data 2020',
                'description': 'Demographic and population statistics from the 2020 US Census for Nebraska counties and census tracts. Includes total population, age distribution, race and ethnicity data.',
                'snippet': 'Census population demographics for Nebraska',
                'tags': ['census', 'population', 'demographics', 'nebraska', '2020'],
                'owner': 'UNL Libraries',
                'created': '1640995200000',
                'modified': '1672531200000',
                'type': 'Feature Service',
                'numViews': 1250,
                'thumbnail': 'census_thumb.png'
            },
            {
                'id': 'sample_agri_002', 
                'title': 'Nebraska Agricultural Land Use',
                'description': 'Comprehensive agricultural land use data for Nebraska including crop types, irrigation patterns, and land cover classifications. Data collected from satellite imagery and field surveys.',
                'snippet': 'Agricultural land use and crop data',
                'tags': ['agriculture', 'land use', 'crops', 'farming', 'nebraska'],
                'owner': 'UNL Extension',
                'created': '1609459200000',
                'modified': '1675123200000',
                'type': 'Map Service',
                'numViews': 890,
                'thumbnail': 'agri_thumb.png'
            },
            {
                'id': 'sample_elevation_003',
                'title': 'Nebraska Digital Elevation Model (DEM)',
                'description': 'High-resolution digital elevation model covering the state of Nebraska. 10-meter resolution DEM derived from LiDAR data, suitable for topographic analysis and watershed modeling.',
                'snippet': 'Digital elevation model topography data',
                'tags': ['elevation', 'DEM', 'topography', 'lidar', 'terrain'],
                'owner': 'UNL Geography',
                'created': '1577836800000',
                'modified': '1671667200000',
                'type': 'Image Service',
                'numViews': 2100,
                'thumbnail': 'dem_thumb.png'
            },
            {
                'id': 'sample_water_004',
                'title': 'Nebraska Water Resources',
                'description': 'Water resources data including rivers, lakes, reservoirs, and groundwater information. Includes water quality measurements and flow data for major water bodies.',
                'snippet': 'Water resources and hydrology data',
                'tags': ['water', 'hydrology', 'rivers', 'lakes', 'groundwater'],
                'owner': 'UNL Water Sciences',
                'created': '1596240000000',
                'modified': '1673740800000',
                'type': 'Feature Service',
                'numViews': 750,
                'thumbnail': 'water_thumb.png'
            },
            {
                'id': 'sample_transport_005',
                'title': 'Nebraska Transportation Networks',
                'description': 'Comprehensive transportation network data including roads, highways, railroads, and airports. Road data includes functional classification and traffic volume where available.',
                'snippet': 'Transportation and road network data',
                'tags': ['transportation', 'roads', 'highways', 'railroads', 'infrastructure'],
                'owner': 'UNL Engineering',
                'created': '1588291200000',
                'modified': '1674950400000',
                'type': 'Feature Service',
                'numViews': 640,
                'thumbnail': 'transport_thumb.png'
            }
        ]
        
        return sample_data

# Free LLM Integration using Hugging Face Inference API
class FreeLLMClient:
    """Free LLM client using Hugging Face Inference API"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 30
        
        # Multiple free models to try (no API key required)
        self.models = [
            "microsoft/DialoGPT-medium",
            "facebook/blenderbot-400M-distill",
            "microsoft/DialoGPT-small",
            "gpt2",
        ]
        
        # Hugging Face Inference API endpoint
        self.hf_api_base = "https://api-inference.huggingface.co/models/"
        
        self.working_model = None
        self.find_working_model()
    
    def find_working_model(self):
        """Find a working free model"""
        st.sidebar.write("**🤖 Testing free AI models...**")
        
        for model in self.models:
            if self.test_model(model):
                self.working_model = model
                st.sidebar.success(f"✅ Using model: {model}")
                return
        
        st.sidebar.warning("⚠️ No AI models available, using fallback method")
        self.working_model = None
    
    def test_model(self, model_name: str) -> bool:
        """Test if a model is available"""
        try:
            url = f"{self.hf_api_base}{model_name}"
            response = self.session.post(
                url,
                json={"inputs": "test"},
                timeout=10
            )
            return response.status_code in [200, 503]  # 503 means model is loading
        except:
            return False
    
    def enhance_query(self, query: str) -> str:
        """Enhance search query using free LLM or fallback method"""
        if not self.working_model:
            return self.fallback_query_enhancement(query)
        
        try:
            url = f"{self.hf_api_base}{self.working_model}"
            
            # Create a prompt for query expansion
            prompt = f"Expand this search query with related geographic and data terms: '{query}'. Related terms:"
            
            response = self.session.post(
                url,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 50,
                        "temperature": 0.3,
                        "do_sample": True
                    }
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')
                    # Extract the expansion part
                    if 'Related terms:' in generated_text:
                        expansion = generated_text.split('Related terms:')[-1].strip()
                        return f"{query} {expansion}".strip()
                elif isinstance(result, dict) and 'generated_text' in result:
                    generated_text = result['generated_text']
                    if 'Related terms:' in generated_text:
                        expansion = generated_text.split('Related terms:')[-1].strip()
                        return f"{query} {expansion}".strip()
                        
        except Exception as e:
            st.sidebar.warning(f"AI enhancement failed: {str(e)[:50]}...")
        
        return self.fallback_query_enhancement(query)
    
    def fallback_query_enhancement(self, query: str) -> str:
        """Fallback query enhancement using predefined mappings"""
        query_lower = query.lower()
        
        # Predefined expansion mappings for common GIS terms
        expansions = {
            'population': 'census demographics people residents statistics household',
            'people': 'population census demographics residents statistics',
            'census': 'population demographics people statistics household income',
            'agriculture': 'farming crops agricultural land use cultivation soil',
            'farming': 'agriculture crops agricultural land use cultivation',
            'water': 'hydrology rivers lakes streams watersheds precipitation',
            'elevation': 'topography terrain DEM digital elevation model height',
            'transportation': 'roads highways infrastructure traffic network',
            'environment': 'ecology natural resources conservation wildlife',
            'climate': 'weather temperature precipitation meteorology',
            'economics': 'economic business industry commerce employment',
            'housing': 'residential buildings construction real estate',
            'education': 'schools university college academic institutions',
            'health': 'healthcare medical hospitals public health',
            'boundaries': 'administrative political jurisdictional borders',
            'land use': 'zoning development urban planning property',
        }
        
        enhanced_terms = [query]
        
        for key, expansion in expansions.items():
            if key in query_lower:
                enhanced_terms.append(expansion)
                break
        
        return ' '.join(enhanced_terms)
    
    def chat_with_user(self, user_message: str, search_results: List) -> str:
        """Chat with user about search results"""
        # Prepare context about search results
        results_context = ""
        if search_results:
            results_context = f"Found {len(search_results)} datasets:\n"
            for i, result in enumerate(search_results[:3]):
                item = result['item']
                results_context += f"{i+1}. {item.title} ({item.type})\n"
        
        # Simple rule-based responses for common questions
        message_lower = user_message.lower()
        
        if 'population' in message_lower or 'census' in message_lower or 'people' in message_lower:
            return f"I can help you find population and demographic data! {results_context}Look for datasets with 'census', 'demographics', or 'population' in their titles. These typically include age distributions, household statistics, and population counts by geographic area."
        
        elif 'agriculture' in message_lower or 'farming' in message_lower or 'crop' in message_lower:
            return f"For agricultural data, I'd recommend looking for: {results_context}• Crop type and land use datasets\n• Agricultural statistics\n• Soil and climate data\n• Irrigation and water use information"
        
        elif 'water' in message_lower or 'river' in message_lower or 'lake' in message_lower:
            return f"Water resources data might include: {results_context}• Hydrological networks (rivers, streams, lakes)\n• Water quality measurements\n• Watershed boundaries\n• Precipitation and climate data"
        
        elif 'elevation' in message_lower or 'topography' in message_lower or 'terrain' in message_lower:
            return f"For elevation and terrain data: {results_context}• Digital Elevation Models (DEMs)\n• Topographic maps\n• LiDAR-derived datasets\n• Slope and aspect calculations"
        
        else:
            return f"I can help you explore the available GIS datasets! {results_context}Try searching for terms like 'population', 'agriculture', 'water', 'elevation', or 'transportation'. You can also browse by data type or owner to discover relevant datasets."

def setup_free_llm():
    """Setup free LLM client"""
    try:
        client = FreeLLMClient()
        return True, client
    except Exception as e:
        st.sidebar.error(f"LLM setup failed: {e}")
        return False, None

@st.cache_resource
def load_tfidf_vectorizer():
    """Load TF-IDF vectorizer"""
    try:
        return TfidfVectorizer(
            max_features=5000,
            stop_words=['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'],
            ngram_range=(1, 2),
            lowercase=True,
            min_df=1,
            max_df=0.9
        )
    except Exception as e:
        st.error(f"Failed to load vectorizer: {e}")
        return None

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def load_portal_data(group_id: str = None):
    """Load and process data from portal with optional group filtering"""
    portal = ArcGISPortalConnector()
    
    with st.spinner("Loading data from UNL Geoportal..."):
        raw_items = portal.get_all_content(max_items=1000, group_id=group_id)
    
    if not raw_items:
        st.error("No data could be loaded from the portal")
        return []
    
    # Convert to GISDataItem objects
    data_items = []
    for item in raw_items:
        try:
            item_id = item.get('id', f"unknown_{len(data_items)}")
            
            # Handle different date formats
            def safe_date_convert(date_value):
                if not date_value:
                    return ''
                try:
                    if isinstance(date_value, (int, float)):
                        # Convert timestamp (milliseconds) to string
                        from datetime import datetime
                        return datetime.fromtimestamp(date_value / 1000).isoformat()
                    return str(date_value)
                except:
                    return str(date_value) if date_value else ''
            
            gis_item = GISDataItem(
                id=item_id,
                title=item.get('title', 'Untitled Dataset'),
                description=item.get('description', '') or item.get('snippet', ''),
                tags=item.get('tags', []) if isinstance(item.get('tags'), list) else [],
                owner=item.get('owner', 'Unknown'),
                created=safe_date_convert(item.get('created')),
                modified=safe_date_convert(item.get('modified')),
                type=item.get('type', 'Dataset'),
                url=f"https://geoportal.unl.edu/portal/home/item.html?id={item_id}",
                portal_url=f"https://geoportal.unl.edu/portal/apps/sites/#/unl-geoportal/datasets/{item_id}",
                thumbnail=item.get('thumbnail'),
                extent=item.get('extent'),
                num_views=item.get('numViews', 0) if isinstance(item.get('numViews'), (int, float)) else 0,
                snippet=item.get('snippet', '')
            )
            data_items.append(gis_item)
            
        except Exception as e:
            logger.warning(f"Error processing item: {e}")
            continue
    
    return data_items

@st.cache_data(ttl=1800)
def create_search_index(data_items):
    """Create search index from data items"""
    vectorizer = load_tfidf_vectorizer()
    if not vectorizer or not data_items:
        return None, None, None
    
    # Create searchable text focusing on title and description
    texts = []
    for item in data_items:
        text_parts = []
        
        # Title (high importance)
        if item.title:
            text_parts.append(item.title)
        
        # Description (high importance - as you requested)
        description = item.description or item.snippet or ""
        if description:
            clean_desc = re.sub(r'<[^>]+>', '', description)  # Remove HTML
            clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()  # Clean whitespace
            text_parts.append(clean_desc)
        
        # Tags (medium importance)
        if item.tags and isinstance(item.tags, list):
            try:
                clean_tags = [str(tag) for tag in item.tags if tag and str(tag).strip()]
                if clean_tags:
                    text_parts.append(" ".join(clean_tags))
            except:
                pass  # Skip problematic tags
        
        # Type (low importance)
        if item.type:
            text_parts.append(item.type)
        
        # Combine all text
        combined_text = " ".join([part for part in text_parts if part.strip()])
        
        if not combined_text.strip():
            combined_text = f"dataset {item.type or 'unknown'} {item.id}"
        
        texts.append(combined_text)
    
    # Create TF-IDF matrix
    try:
        with st.spinner("Building search index..."):
            tfidf_matrix = vectorizer.fit_transform(texts)
            
        vocab_size = len(vectorizer.vocabulary_)
        st.sidebar.success(f"✅ Search index ready: {vocab_size} terms, {len(texts)} documents")
        
        return vectorizer, tfidf_matrix, texts
        
    except Exception as e:
        st.error(f"Failed to create search index: {e}")
        return None, None, None

def semantic_search(query: str, data_items: List[GISDataItem], vectorizer, tfidf_matrix, texts, top_k: int = 10, use_ai: bool = False, llm_client=None):
    """Perform semantic search using TF-IDF"""
    if not vectorizer or tfidf_matrix is None:
        return []
    
    try:
        # Enhance query with AI if available
        search_query = query
        if use_ai and llm_client:
            search_query = llm_client.enhance_query(query)
            st.sidebar.write(f"**Enhanced query:** {search_query}")
        
        # Create query vector
        query_vector = vectorizer.transform([search_query.lower()])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Low threshold
                item = data_items[idx]
                results.append({
                    'item': item,
                    'similarity': float(similarities[idx]),
                    'text': texts[idx] if texts else "",
                    'matched_content': texts[idx][:300] if texts else ""
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
        
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

def keyword_search(query: str, data_items: List[GISDataItem], top_k: int = 10):
    """Enhanced keyword search focusing on title and description"""
    query_lower = query.lower()
    query_words = re.findall(r'\w+', query_lower)
    
    results = []
    for item in data_items:
        score = 0
        
        title = (item.title or "").lower()
        description = (item.description or item.snippet or "").lower()
        tags = " ".join(item.tags or []).lower()
        
        # Scoring with emphasis on title and description
        for word in query_words:
            if word in title:
                score += 5
            if word in description:
                score += 5
            if word in tags:
                score += 2
        
        # Phrase matching bonus
        if query_lower in description:
            score += 15
        elif query_lower in title:
            score += 10
        
        if score > 0:
            matched_content = f"Title: {item.title}\nDescription: {description[:250]}..."
            results.append({
                'item': item,
                'similarity': score / max(len(query_words), 1),
                'text': f"{title} {description}",
                'matched_content': matched_content
            })
    
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_k]

def display_search_result(result, index):
    """Display a single search result"""
    item = result['item']
    similarity = result['similarity']
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Title with links
            st.markdown(f"### {item.title}")
            
            # Links
            link_col1, link_col2 = st.columns(2)
            with link_col1:
                st.markdown(f"🔗 [View Details]({item.url})")
            with link_col2:
                st.markdown(f"📊 [Explore Dataset]({item.portal_url})")
            
            # Type and metadata
            st.markdown(f"**Type:** `{item.type}`")
            
            # Description (emphasis as requested)
            description = item.description or item.snippet or "No description available"
            if description:
                st.write("**Description:**")
                st.write(description[:500] + "..." if len(description) > 500 else description)
            
            # Matched content
            if 'matched_content' in result and result['matched_content']:
                with st.expander("🎯 Matched content"):
                    st.write(result['matched_content'])
            
            # Tags
            if item.tags and isinstance(item.tags, list):
                try:
                    tags_html = " ".join([f'<span style="background-color: #e6f3ff; padding: 2px 6px; border-radius: 3px; font-size: 12px; margin: 2px;">{str(tag)}</span>' for tag in item.tags[:8] if tag])
                    st.markdown(f"**Tags:** {tags_html}", unsafe_allow_html=True)
                except Exception as e:
                    st.caption(f"**Tags:** {', '.join([str(tag) for tag in item.tags[:8] if tag])}")
            
            # Metadata row
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            with meta_col1:
                st.caption(f"👤 Owner: {item.owner}")
            with meta_col2:
                try:
                    if item.num_views and isinstance(item.num_views, (int, float)) and item.num_views > 0:
                        st.caption(f"👁️ Views: {int(item.num_views)}")
                    else:
                        st.caption("👁️ Views: N/A")
                except:
                    st.caption("👁️ Views: N/A")
            with meta_col3:
                # Handle different date formats (timestamp vs string)
                modified_date = "Unknown"
                if item.modified:
                    try:
                        if isinstance(item.modified, (int, float)):
                            # Convert timestamp to readable date
                            from datetime import datetime
                            modified_date = datetime.fromtimestamp(item.modified / 1000).strftime('%Y-%m-%d')
                        elif isinstance(item.modified, str) and len(item.modified) >= 10:
                            modified_date = item.modified[:10]
                        else:
                            modified_date = str(item.modified)
                    except:
                        modified_date = "Unknown"
                st.caption(f"📅 Modified: {modified_date}")
        
        with col2:
            # Match score
            st.metric("Match Score", f"{similarity:.1%}")
            
            # Thumbnail placeholder
            if item.thumbnail:
                st.write("📷 Thumbnail available")
            else:
                st.write("📄 No preview")
        
        st.divider()

def main():
    """Main application"""
    
    # Header
    st.title("🗺️ UNL Geoportal Intelligent Search")
    st.markdown("Search for geospatial data using natural language queries powered by **free AI models**!")
    
    # Sidebar
    with st.sidebar:
        st.header("🔒 Content Filter")
        
        # Group filter option
        use_group_filter = st.checkbox(
            "Restrict to UNL Geoportal Content", 
            value=True,  # Default to enabled
            help="Only show datasets shared with the 'UNL Geoportal Content' group"
        )
        
        if use_group_filter:
            # Show both options for flexibility
            filter_method = st.radio(
                "Filter Method:",
                ["Use Group ID (Recommended)", "Use Group Name"],
                help="Group ID is more reliable for filtering"
            )
            
            if filter_method == "Use Group ID (Recommended)":
                group_id = st.text_input(
                    "Group ID", 
                    value="3df944fa3e1c4ad29f11b04cfc6a26a2",  # Your actual group ID
                    help="Enter the group ID from the URL (more reliable)"
                )
                group_name = None
            else:
                group_name = st.text_input(
                    "Group Name", 
                    value="UNL Geoportal Content",
                    help="Enter the exact group name"
                )
                group_id = None
        else:
            group_id = None
            group_name = None
        
        st.header("🤖 Free AI Assistant")
        has_llm, llm_client = setup_free_llm()
        
        if has_llm:
            st.success("✅ Free AI features available")
            
            # Chat interface
            st.subheader("💬 Ask me about GIS data")
            user_message = st.text_area("What can I help you find?", placeholder="e.g., I'm interested in population data for my research")
        else:
            st.info("💡 Free AI models loading...")
            user_message = None
        
        st.header("🔧 Settings")
        search_method = st.selectbox(
            "Search Method",
            ["AI-Enhanced Search", "Semantic Search", "Keyword Search"],
            help="AI-Enhanced uses free LLMs to expand your query"
        )
        
        num_results = st.slider("Number of Results", 5, 25, 10)
        
        st.header("📊 System Status")
    
    # Load data
    try:
        # Use group ID if selected, otherwise use group name
        filter_value = group_id if use_group_filter and group_id else None
        data_items = load_portal_data(group_id=filter_value)
        
        if not data_items:
            if use_group_filter:
                filter_desc = f"group ID '{group_id}'" if group_id else f"group name '{group_name}'"
                st.error(f"❌ No data found for {filter_desc}. Please check the filter settings.")
            else:
                st.error("❌ No data available. Please check the API connection.")
            st.info("💡 Try adjusting the group filter settings in the sidebar.")
            st.stop()
        
        if use_group_filter:
            filter_desc = f"group ID '{group_id}'" if group_id else f"group name '{group_name}'"
            st.sidebar.success(f"✅ {len(data_items)} datasets loaded (filtered by {filter_desc})")
        else:
            st.sidebar.success(f"✅ {len(data_items)} datasets loaded")
        
        # Create search index for semantic/AI search
        vectorizer, tfidf_matrix, texts = None, None, None
        if search_method in ["Semantic Search", "AI-Enhanced Search"]:
            vectorizer, tfidf_matrix, texts = create_search_index(data_items)
            
            if vectorizer is None:
                st.sidebar.warning("⚠️ Falling back to keyword search")
                search_method = "Keyword Search"
        
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.info("💡 Try running the diagnostic tool to identify issues.")
        st.stop()
    
    # Search interface
    st.header("🔍 Natural Language Search")
    
    # Check for query parameter from example buttons
    query_from_params = st.query_params.get("q", "")
    
    query = st.text_input(
        "Describe what geospatial data you're looking for:",
        value=query_from_params,
        placeholder="e.g., I'm interested in population data, Show me agricultural datasets, I need elevation maps...",
        help="Use natural language - the AI will help interpret your request!"
    )
    
    # Handle chat messages in sidebar
    if user_message and user_message.strip() and has_llm:
        with st.sidebar:
            st.subheader("💬 AI Response")
            # Get current search results for context
            current_results = []
            if query and query.strip():
                if search_method == "AI-Enhanced Search" and vectorizer is not None:
                    current_results = semantic_search(query, data_items, vectorizer, tfidf_matrix, texts, 
                                                    5, use_ai=True, llm_client=llm_client)
                elif search_method == "Semantic Search" and vectorizer is not None:
                    current_results = semantic_search(query, data_items, vectorizer, tfidf_matrix, texts, 5)
                else:
                    current_results = keyword_search(query, data_items, 5)
            
            response = llm_client.chat_with_user(user_message, current_results)
            st.write(response)
    
    # Perform search
    if query and query.strip():
        with st.spinner("🤖 AI is analyzing your request and searching..."):
            
            if search_method == "AI-Enhanced Search" and vectorizer is not None:
                results = semantic_search(query, data_items, vectorizer, tfidf_matrix, texts, 
                                        num_results, use_ai=True, llm_client=llm_client)
            elif search_method == "Semantic Search" and vectorizer is not None:
                results = semantic_search(query, data_items, vectorizer, tfidf_matrix, texts, 
                                        num_results, use_ai=False)
            else:
                results = keyword_search(query, data_items, num_results)
            
        # Display results
        if results:
            st.success(f"🎯 Found {len(results)} relevant datasets")
            
            # Show AI enhancement info if used
            if search_method == "AI-Enhanced Search" and has_llm:
                st.info("🤖 Your query was enhanced using free AI models to find more relevant results!")
            
            # Export functionality
            if results:
                results_df = pd.DataFrame([{
                    'Title': r['item'].title,
                    'Description': (r['item'].description or r['item'].snippet or "")[:200],
                    'Type': r['item'].type,
                    'Owner': r['item'].owner,
                    'Portal_URL': r['item'].portal_url,
                    'Tags': ', '.join(r['item'].tags),
                    'Match_Score': f"{r['similarity']:.3f}"
                } for r in results])
                
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Export Results (CSV)",
                    data=csv_data,
                    file_name=f"unl_geoportal_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Display each result
            for i, result in enumerate(results):
                display_search_result(result, i)
                
        else:
            st.warning("No results found. Try different keywords or broader terms.")
            st.info("💡 Try natural language like: 'population data', 'agricultural information', 'water resources'")
    
    # Quick search examples
    st.header("💡 Try These Natural Language Searches")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗣️ Natural Language Examples")
        example_queries_nl = [
            "I'm interested in population data",
            "Show me agricultural datasets", 
            "I need elevation information",
            "Find water resource data"
        ]
        
        for i, example_query in enumerate(example_queries_nl):
            if st.button(f'"{example_query}"', key=f"nl_example_{i}"):
                st.query_params["q"] = example_query
                st.rerun()
    
    with col2:
        st.subheader("🔍 Keyword Examples")
        example_queries_kw = [
            "census demographics nebraska",
            "agriculture farming crops",
            "DEM elevation topography", 
            "rivers lakes hydrology"
        ]
        
        for i, example_query in enumerate(example_queries_kw):
            if st.button(f'"{example_query}"', key=f"kw_example_{i}"):
                st.query_params["q"] = example_query
                st.rerun()
    
    # Show AI capabilities
    st.header("🤖 Free AI Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Query Enhancement**\nAI expands your search terms with related geographic concepts")
    
    with col2:
        st.info("**Natural Language**\nType requests like 'I need population data' instead of keywords")
    
    with col3:
        st.info("**Smart Chat**\nAsk questions about available datasets and get helpful suggestions")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏛️ <a href='https://geoportal.unl.edu/portal/apps/sites/#/unl-geoportal' target='_blank'>UNL Geoportal</a> | 
        🤖 Powered by Free AI Models | 
        🔍 Intelligent Search System | 
        📚 <a href='https://libraries.unl.edu/' target='_blank'>UNL Libraries</a></p>
        <p><small>🆓 No API keys required - Uses free Hugging Face models</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()