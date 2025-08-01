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
    
    def search_content(self, query: str = "*", num: int = 100, start: int = 1) -> Dict:
        """Search for content using the working endpoint"""
        if not self.working_endpoint:
            return {'results': [], 'total': 0}
        
        # Use the working parameters as base
        params = self.working_params.copy()
        
        # Update with search parameters
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
    
    def get_all_content(self, max_items: int = 1000) -> List[Dict]:
        """Get all available content with progress tracking"""
        all_items = []
        start = 1
        num_per_request = 50  # Smaller batches for better reliability
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
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
                    start=current_start
                )
                
                if not result['results']:
                    break
                
                items_from_this_search.extend(result['results'])
                
                # Update progress
                progress = min(len(items_from_this_search) / max_items, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Found {len(items_from_this_search)} items with '{search_term}'")
                
                # Check if we got fewer results than requested (end of results)
                if len(result['results']) < num_per_request:
                    break
                    
                current_start += num_per_request
                time.sleep(0.2)  # Be nice to the server
            
            if items_from_this_search:
                all_items = items_from_this_search
                st.sidebar.success(f"✅ Loaded {len(all_items)} items using search term: '{search_term}'")
                break
        
        # If still no results, try sample data
        if not all_items:
            st.warning("⚠️ Could not load data from API. Using sample data for demonstration.")
            all_items = self.create_sample_data()
        
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

# OpenAI Integration (optional)
def setup_openai():
    """Setup OpenAI client with conditional import"""
    try:
        import openai
        api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password", 
                                      help="Enter your OpenAI API key for AI-enhanced search")
        if api_key:
            openai.api_key = api_key
            return True, openai
        else:
            env_key = os.getenv('OPENAI_API_KEY')
            if env_key:
                openai.api_key = env_key
                return True, openai
    except ImportError:
        st.sidebar.info("💡 Install 'openai' package for AI-enhanced features")
        return False, None
    
    return False, None

def enhance_query_with_ai(query: str, openai_module) -> str:
    """Use OpenAI to enhance search queries"""
    if not openai_module:
        return query
        
    try:
        response = openai_module.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a GIS expert. Expand search queries with relevant geospatial terms.

Examples:
- "population" → "population demographics census people residents statistics"
- "water" → "water hydrology rivers lakes streams watersheds"
- "agriculture" → "agriculture farming crops land use cultivation"

Return expanded terms in one line."""},
                {"role": "user", "content": f"Expand: {query}"}
            ],
            max_tokens=50,
            temperature=0.3
        )
        
        expanded = response.choices[0].message.content.strip()
        return f"{query} {expanded}"
        
    except Exception as e:
        st.sidebar.warning(f"AI enhancement failed: {e}")
        return query

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
def load_portal_data():
    """Load and process data from portal"""
    portal = ArcGISPortalConnector()
    
    with st.spinner("Loading data from UNL Geoportal..."):
        raw_items = portal.get_all_content(max_items=1000)
    
    if not raw_items:
        st.error("No data could be loaded from the portal")
        return []
    
    # Convert to GISDataItem objects
    data_items = []
    for item in raw_items:
        try:
            item_id = item.get('id', f"unknown_{len(data_items)}")
            
            gis_item = GISDataItem(
                id=item_id,
                title=item.get('title', 'Untitled Dataset'),
                description=item.get('description', '') or item.get('snippet', ''),
                tags=item.get('tags', []),
                owner=item.get('owner', 'Unknown'),
                created=item.get('created', ''),
                modified=item.get('modified', ''),
                type=item.get('type', 'Dataset'),
                url=f"https://geoportal.unl.edu/portal/home/item.html?id={item_id}",
                portal_url=f"https://geoportal.unl.edu/portal/apps/sites/#/unl-geoportal/datasets/{item_id}",
                thumbnail=item.get('thumbnail'),
                extent=item.get('extent'),
                num_views=item.get('numViews', 0),
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
        if item.tags:
            text_parts.append(" ".join(item.tags))
        
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

def semantic_search(query: str, data_items: List[GISDataItem], vectorizer, tfidf_matrix, texts, top_k: int = 10, use_ai: bool = False, openai_module=None):
    """Perform semantic search using TF-IDF"""
    if not vectorizer or tfidf_matrix is None:
        return []
    
    try:
        # Enhance query with AI if available
        search_query = query
        if use_ai and openai_module:
            search_query = enhance_query_with_ai(query, openai_module)
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
            if item.tags:
                tags_html = " ".join([f'<span style="background-color: #e6f3ff; padding: 2px 6px; border-radius: 3px; font-size: 12px; margin: 2px;">{tag}</span>' for tag in item.tags[:8]])
                st.markdown(f"**Tags:** {tags_html}", unsafe_allow_html=True)
            
            # Metadata row
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            with meta_col1:
                st.caption(f"👤 Owner: {item.owner}")
            with meta_col2:
                if item.num_views:
                    st.caption(f"👁️ Views: {item.num_views}")
            with meta_col3:
                st.caption(f"📅 Modified: {item.modified[:10] if item.modified else 'Unknown'}")
        
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
    st.markdown("Search for geospatial data using natural language queries!")
    
    # Sidebar
    with st.sidebar:
        st.header("🤖 AI Assistant")
        has_openai, openai_module = setup_openai()
        
        if has_openai:
            st.success("✅ AI features available")
        else:
            st.info("💡 Add OpenAI API key for enhanced search")
        
        st.header("🔧 Settings")
        search_method = st.selectbox(
            "Search Method",
            ["Keyword Search", "Semantic Search", "AI-Enhanced Search"],
            help="Choose your search strategy"
        )
        
        num_results = st.slider("Number of Results", 5, 25, 10)
        
        st.header("📊 System Status")
    
    # Load data
    try:
        data_items = load_portal_data()
        
        if not data_items:
            st.error("❌ No data available. Please check the API connection.")
            st.info("💡 The diagnostic tool can help identify connection issues.")
            st.stop()
        
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
    st.header("🔍 Search Interface")
    
    query = st.text_input(
        "What geospatial data are you looking for?",
        placeholder="e.g., population census data, agricultural land use, elevation maps...",
        help="Describe what you need - the search looks through titles and descriptions"
    )
    
    # Perform search
    if query and query.strip():
        with st.spinner("Searching..."):
            
            if search_method == "AI-Enhanced Search" and vectorizer is not None:
                results = semantic_search(query, data_items, vectorizer, tfidf_matrix, texts, 
                                        num_results, use_ai=has_openai, openai_module=openai_module)
            elif search_method == "Semantic Search" and vectorizer is not None:
                results = semantic_search(query, data_items, vectorizer, tfidf_matrix, texts, 
                                        num_results, use_ai=False)
            else:
                results = keyword_search(query, data_items, num_results)
            
        # Display results
        if results:
            st.success(f"Found {len(results)} relevant datasets")
            
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
            st.info("💡 Try searches like: 'population', 'agriculture', 'water', 'elevation'")
    
    # Quick search examples
    st.header("💡 Example Searches")
    
    col1, col2, col3, col4 = st.columns(4)
    
    example_queries = [
        ("🏛️ Population", "population census demographics"),
        ("🌾 Agriculture", "agriculture farming crops land use"),
        ("💧 Water", "water hydrology rivers lakes"),
        ("🏔️ Elevation", "elevation topography DEM terrain")
    ]
    
    for i, (button_text, example_query) in enumerate(example_queries):
        col = [col1, col2, col3, col4][i]
        with col:
            if st.button(button_text, key=f"example_{i}"):
                st.experimental_set_query_params(q=example_query)
                st.experimental_rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏛️ <a href='https://geoportal.unl.edu/portal/apps/sites/#/unl-geoportal' target='_blank'>UNL Geoportal</a> | 
        🔍 Intelligent Search System | 
        📚 <a href='https://libraries.unl.edu/' target='_blank'>UNL Libraries</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()