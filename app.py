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
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

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
    portal_url: str  # Link to the geoportal site
    thumbnail: Optional[str] = None
    extent: Optional[Dict] = None
    num_views: Optional[int] = None
    snippet: Optional[str] = None

class ArcGISPortalConnector:
    """Handles connection and data retrieval from ArcGIS Portal"""
    
    def __init__(self, portal_url: str):
        # Try different possible API endpoints
        self.base_portal_url = "https://geoportal.unl.edu/portal"
        self.api_endpoints = [
            f"{self.base_portal_url}/sharing/rest",
            f"{self.base_portal_url}/rest/services",
            "https://geoportal.unl.edu/sharing/rest",
            "https://geoportal.unl.edu/rest/search"
        ]
        self.session = requests.Session()
        self.session.timeout = 30
        self.working_endpoint = None
        
        # Test endpoints to find the working one
        self.find_working_endpoint()
    
    def find_working_endpoint(self):
        """Test different API endpoints to find the working one"""
        for endpoint in self.api_endpoints:
            try:
                # Test with a simple query
                test_urls = [
                    f"{endpoint}/search?q=*&f=json&num=1",
                    f"{endpoint}?f=json",
                    f"{endpoint}/content/items?f=json&num=1"
                ]
                
                for test_url in test_urls:
                    try:
                        response = self.session.get(test_url, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            if 'results' in data or 'items' in data or 'services' in data:
                                self.working_endpoint = endpoint
                                st.sidebar.success(f"✅ Connected to: {endpoint}")
                                return
                    except:
                        continue
                        
            except:
                continue
        
        st.sidebar.error("❌ Could not connect to any API endpoint")
    
    def search_content(self, query: str = "*", num: int = 100, start: int = 1) -> Dict:
        """Search for content in the portal using multiple methods"""
        if not self.working_endpoint:
            return {'results': [], 'total': 0}
        
        # Try different search URLs
        search_urls = [
            f"{self.working_endpoint}/search",
            f"{self.working_endpoint}/content/items",
            f"{self.base_portal_url}/sharing/rest/search"
        ]
        
        params = {
            'q': query,
            'num': min(num, 100),
            'start': start,
            'f': 'json',
            'sortField': 'modified',
            'sortOrder': 'desc'
        }
        
        for search_url in search_urls:
            try:
                response = self.session.get(search_url, params=params, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    
                    # Handle different response formats
                    if 'results' in result:
                        return result
                    elif 'items' in result:
                        return {'results': result['items'], 'total': len(result['items'])}
                    elif isinstance(result, list):
                        return {'results': result, 'total': len(result)}
                        
            except Exception as e:
                logger.warning(f"Search failed for {search_url}: {e}")
                continue
        
        return {'results': [], 'total': 0}
    
    def get_all_content(self, max_items: int = 1000) -> List[Dict]:
        """Get all available content from the portal"""
        all_items = []
        start = 1
        num_per_request = 100
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Try different search strategies
        search_queries = ["*", "type:*", ""]
        
        for query in search_queries:
            if all_items:  # If we already got data, break
                break
                
            current_items = []
            current_start = 1
            
            while len(current_items) < max_items:
                status_text.text(f"Loading data with query '{query}'... {len(current_items)} items retrieved")
                
                result = self.search_content(query=query, num=num_per_request, start=current_start)
                
                if 'results' not in result or not result['results']:
                    break
                    
                current_items.extend(result['results'])
                
                # Update progress
                if 'total' in result and result['total'] > 0:
                    progress = min(len(current_items) / min(result['total'], max_items), 1.0)
                    progress_bar.progress(progress)
                
                # Check if we've got all available items
                if len(result['results']) < num_per_request:
                    break
                    
                current_start += num_per_request
                time.sleep(0.1)
            
            if current_items:
                all_items = current_items
                break
        
        # If still no results, try to get data from a known working ArcGIS Portal endpoint
        if not all_items:
            try:
                # Try the standard ArcGIS Online search
                fallback_url = f"{self.base_portal_url}/sharing/rest/search"
                params = {'q': '*', 'f': 'json', 'num': 100}
                
                response = self.session.get(fallback_url, params=params, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    if 'results' in result:
                        all_items = result['results']
                        
            except Exception as e:
                logger.error(f"Fallback search failed: {e}")
        
        progress_bar.progress(1.0)
        status_text.text(f"Loaded {len(all_items)} items successfully!")
        
        return all_items[:max_items]

# OpenAI Integration
def setup_openai():
    """Setup OpenAI client"""
    api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password", help="Enter your OpenAI API key for enhanced search")
    if api_key:
        openai.api_key = api_key
        return True
    else:
        # Try to get from environment
        env_key = os.getenv('OPENAI_API_KEY')
        if env_key:
            openai.api_key = env_key
            return True
    return False

def enhance_query_with_ai(query: str) -> str:
    """Use OpenAI to enhance search queries"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a GIS and geospatial data expert. Given a user's search query, expand it with relevant GIS terms and concepts that would help find geospatial datasets.

For example:
- "population" → "population demographics census people residents statistics inhabitants"
- "water" → "water hydrology rivers lakes streams watersheds precipitation"
- "agriculture" → "agriculture farming crops land use cultivation"
- "elevation" → "elevation topography DEM digital elevation model terrain"

Return only the expanded search terms as a single line, separated by spaces."""},
                {"role": "user", "content": f"Expand this geospatial search query: {query}"}
            ],
            max_tokens=100,
            temperature=0.3
        )
        
        expanded = response.choices[0].message.content.strip()
        return f"{query} {expanded}"
        
    except Exception as e:
        st.sidebar.warning(f"AI enhancement failed: {e}")
        return query

def chat_with_user(user_message: str, search_results: List) -> str:
    """Chat with user about search results using OpenAI"""
    try:
        # Prepare context about the search results
        results_context = ""
        if search_results:
            results_context = "Available datasets found:\n"
            for i, result in enumerate(search_results[:5]):
                item = result['item']
                results_context += f"{i+1}. {item.title} - {item.type}\n"
                if item.description:
                    results_context += f"   Description: {item.description[:100]}...\n"
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"""You are a helpful GIS data assistant for the University of Nebraska-Lincoln Geoportal. 
                
Help users find and understand geospatial data. Be conversational and helpful.

Current search context: {results_context}

Guidelines:
- Provide specific recommendations about the datasets found
- Explain what types of GIS data might be useful for their needs  
- Suggest related searches if current results aren't perfect
- Be encouraging and helpful"""},
                {"role": "user", "content": user_message}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"I'd be happy to help you find GIS data! However, I'm having trouble accessing my AI capabilities right now. Try searching for specific terms related to your research needs."

@st.cache_resource
def load_tfidf_vectorizer():
    """Load the TF-IDF vectorizer (cached)"""
    try:
        # Minimal stop words to preserve GIS terminology
        custom_stop_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        
        return TfidfVectorizer(
            max_features=10000,
            stop_words=custom_stop_words,
            ngram_range=(1, 3),
            lowercase=True,
            min_df=1,
            max_df=0.95,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9_]*\b'
        )
    except Exception as e:
        st.error(f"Failed to load vectorizer: {e}")
        return None

@st.cache_data(ttl=3600)
def load_portal_data():
    """Load and process data from the portal (cached)"""
    portal = ArcGISPortalConnector("https://geoportal.unl.edu/portal/sharing/rest")
    
    with st.spinner("Loading data from UNL Geoportal..."):
        raw_items = portal.get_all_content(max_items=2000)
    
    # Convert to GISDataItem objects
    data_items = []
    for item in raw_items:
        try:
            item_id = item.get('id', '')
            
            # Create different URL formats to try
            portal_urls = [
                f"https://geoportal.unl.edu/portal/apps/sites/#/unl-geoportal/datasets/{item_id}",
                f"https://geoportal.unl.edu/portal/home/item.html?id={item_id}",
                f"https://geoportal.unl.edu/portal/apps/sites/#/unl-geoportal/datasets/{item_id}/explore"
            ]
            
            gis_item = GISDataItem(
                id=item_id,
                title=item.get('title', 'Untitled'),
                description=item.get('description', '') or item.get('snippet', ''),
                tags=item.get('tags', []),
                owner=item.get('owner', 'Unknown'),
                created=item.get('created', ''),
                modified=item.get('modified', ''),
                type=item.get('type', 'Unknown'),
                url=f"https://geoportal.unl.edu/portal/home/item.html?id={item_id}",
                portal_url=f"https://geoportal.unl.edu/portal/apps/sites/#/unl-geoportal/datasets/{item_id}",
                thumbnail=item.get('thumbnail'),
                extent=item.get('extent'),
                num_views=item.get('numViews', 0),
                snippet=item.get('snippet', '')
            )
            data_items.append(gis_item)
        except Exception as e:
            logger.warning(f"Error processing item {item.get('id', 'unknown')}: {e}")
            continue
    
    return data_items

@st.cache_data(ttl=3600)
def create_search_index(data_items):
    """Create TF-IDF search index for all data items (cached)"""
    vectorizer = load_tfidf_vectorizer()
    if not vectorizer:
        return None, None, None
    
    if not data_items:
        st.error("No data items to index!")
        return None, None, None
    
    # Prepare text for indexing - focus on description and context
    texts = []
    for item in data_items:
        text_parts = []
        
        # Title (most important)
        if item.title:
            text_parts.append(item.title)
        
        # Description (very important - this is what you requested)
        description = item.description or item.snippet or ""
        if description:
            # Clean and add full description
            clean_description = re.sub(r'<[^>]+>', '', description)  # Remove HTML tags
            text_parts.append(clean_description)
        
        # Tags (important for categorization)
        if item.tags:
            text_parts.append(" ".join(item.tags))
        
        # Type and owner
        if item.type:
            text_parts.append(item.type)
        if item.owner:
            text_parts.append(item.owner)
        
        # Join all parts
        text = " ".join([part for part in text_parts if part.strip()])
        
        # Ensure we have some text
        if not text.strip():
            text = f"untitled {item.type or 'dataset'} {item.id}"
        
        texts.append(text)
    
    # Debug: Show some sample texts
    st.sidebar.write("**Sample indexed content:**")
    for i, text in enumerate(texts[:3]):
        st.sidebar.write(f"{i+1}. {text[:150]}...")
    
    # Create TF-IDF matrix
    with st.spinner("Creating search index..."):
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Debug info
            vocab_size = len(vectorizer.vocabulary_)
            st.sidebar.success(f"✅ Index created: {vocab_size} terms, {len(texts)} documents")
            
            return vectorizer, tfidf_matrix, texts
        except Exception as e:
            st.error(f"Failed to create search index: {e}")
            st.error(f"Sample texts: {texts[:3]}")
            return None, None, None

def semantic_search(query: str, data_items: List[GISDataItem], vectorizer, tfidf_matrix, texts, top_k: int = 10, use_ai: bool = False):
    """Perform semantic search using TF-IDF with optional AI enhancement"""
    if not vectorizer or tfidf_matrix is None:
        return []
    
    try:
        # Enhance query with AI if available
        search_query = query
        if use_ai:
            search_query = enhance_query_with_ai(query)
            st.sidebar.write(f"**Enhanced query:** {search_query}")
        
        # Create query vector
        query_vector = vectorizer.transform([search_query.lower()])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Debug info
        max_sim = similarities.max() if len(similarities) > 0 else 0
        st.sidebar.write(f"**Max similarity:** {max_sim:.3f}")
        
        # Get top results with lower threshold
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Low threshold
                item = data_items[idx]
                results.append({
                    'item': item,
                    'similarity': float(similarities[idx]),
                    'text': texts[idx] if texts else "",
                    'matched_content': texts[idx][:200] if texts else ""
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
        
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

def keyword_search(query: str, data_items: List[GISDataItem], top_k: int = 10):
    """Enhanced keyword search focusing on descriptions"""
    query_lower = query.lower()
    query_words = re.findall(r'\w+', query_lower)
    
    results = []
    for item in data_items:
        score = 0
        
        # Focus on description/summary as requested
        description = (item.description or item.snippet or "").lower()
        title = (item.title or "").lower()
        tags = " ".join(item.tags or []).lower()
        
        # Weighted scoring
        for word in query_words:
            # Description gets highest weight
            if word in description:
                score += 5
            # Title gets medium weight
            if word in title:
                score += 3
            # Tags get lower weight
            if word in tags:
                score += 2
        
        # Bonus for exact phrase matches
        if query_lower in description:
            score += 10
        elif query_lower in title:
            score += 5
        
        if score > 0:
            results.append({
                'item': item,
                'similarity': score / max(len(query_words), 1),
                'text': description[:200],
                'matched_content': description[:300]
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
            
            # Links to different views
            link_col1, link_col2 = st.columns(2)
            with link_col1:
                st.markdown(f"🔗 [View in Portal]({item.url})")
            with link_col2:
                st.markdown(f"📊 [Explore Dataset]({item.portal_url})")
            
            # Type badge
            st.markdown(f"**Type:** `{item.type}`")
            
            # Description/Summary (as requested)
            description = item.description or item.snippet or "No description available"
            if description:
                # Show more of the description since it's important for context
                full_desc = description[:500] + "..." if len(description) > 500 else description
                st.write("**Description:**")
                st.write(full_desc)
            
            # Matched content preview
            if 'matched_content' in result and result['matched_content']:
                with st.expander("🎯 Matched content preview"):
                    st.write(result['matched_content'])
            
            # Tags
            if item.tags:
                tags_html = " ".join([f'<span style="background-color: #e6f3ff; padding: 2px 6px; border-radius: 3px; font-size: 12px; margin: 2px;">{tag}</span>' for tag in item.tags[:10]])
                st.markdown(f"**Tags:** {tags_html}", unsafe_allow_html=True)
            
            # Metadata
            col_meta1, col_meta2, col_meta3 = st.columns(3)
            with col_meta1:
                st.caption(f"👤 Owner: {item.owner}")
            with col_meta2:
                if item.num_views:
                    st.caption(f"👁️ Views: {item.num_views}")
            with col_meta3:
                st.caption(f"📅 Modified: {item.modified[:10] if item.modified else 'Unknown'}")
        
        with col2:
            # Similarity score
            st.metric("Match Score", f"{similarity:.1%}")
            
            # Thumbnail if available
            if item.thumbnail:
                thumbnail_url = f"https://geoportal.unl.edu/portal/sharing/rest/content/items/{item.id}/info/{item.thumbnail}"
                try:
                    st.image(thumbnail_url, width=150)
                except:
                    st.write("📷 Thumbnail available")
        
        st.divider()

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🗺️ UNL Geoportal Intelligent Search")
    st.markdown("Search for geospatial data using natural language with AI assistance!")
    
    # Sidebar
    with st.sidebar:
        st.header("🤖 AI Assistant")
        has_openai = setup_openai()
        
        if has_openai:
            st.success("✅ ChatGPT integration active")
            
            # Chat interface
            st.subheader("💬 Ask me about GIS data")
            user_message = st.text_area("What can I help you find?", placeholder="e.g., What kind of population data do you have?")
            
        else:
            st.info("💡 Add OpenAI API key for AI chat features")
        
        st.header("🔧 Settings")
        search_method = st.selectbox(
            "Search Method",
            ["AI-Enhanced Search", "Semantic Search", "Keyword Search"],
            help="AI-Enhanced uses ChatGPT to improve your search"
        )
        
        num_results = st.slider("Number of Results", 5, 50, 10)
        
        st.header("📊 Portal Stats")
    
    # Load data
    try:
        data_items = load_portal_data()
        st.sidebar.success(f"✅ {len(data_items)} datasets loaded")
        
        # Show some examples of loaded data
        if data_items:
            st.sidebar.write("**Sample datasets:**")
            for i, item in enumerate(data_items[:3]):
                st.sidebar.write(f"{i+1}. {item.title[:50]}...")
            
            # Check for census data specifically
            census_items = [item for item in data_items if 'census' in item.title.lower() or any('census' in tag.lower() for tag in item.tags)]
            if census_items:
                st.sidebar.success(f"🎯 Found {len(census_items)} census-related datasets")
        
        # Create search index
        vectorizer, tfidf_matrix, texts = None, None, None
        if search_method in ["AI-Enhanced Search", "Semantic Search"]:
            vectorizer, tfidf_matrix, texts = create_search_index(data_items)
            if vectorizer is not None:
                st.sidebar.success("✅ Search index ready")
            else:
                st.sidebar.error("❌ Search index failed, using keyword search")
                search_method = "Keyword Search"
        
    except Exception as e:
        st.error(f"Failed to load portal data: {e}")
        st.stop()
    
    # Search interface
    st.header("🔍 Search")
    
    # Search input
    query = st.text_input(
        "What geospatial data are you looking for?",
        placeholder="e.g., census population data, agricultural land use, elevation maps...",
        help="Describe what you need in plain English. The AI will help find relevant datasets."
    )
    
    # Search and display results
    if query and query.strip():
        with st.spinner("Searching..."):
            # Perform search based on method
            if search_method == "AI-Enhanced Search" and vectorizer is not None:
                results = semantic_search(query, data_items, vectorizer, tfidf_matrix, texts, num_results, use_ai=has_openai)
            elif search_method == "Semantic Search" and vectorizer is not None:
                results = semantic_search(query, data_items, vectorizer, tfidf_matrix, texts, num_results, use_ai=False)
            else:
                results = keyword_search(query, data_items, num_results)
            
            # Display results
            if results:
                st.success(f"Found {len(results)} relevant datasets")
                
                # Export option
                if len(results) > 0:
                    results_df = pd.DataFrame([{
                        'Title': r['item'].title,
                        'Type': r['item'].type,
                        'Owner': r['item'].owner,
                        'Portal_URL': r['item'].portal_url,
                        'Description': (r['item'].description or r['item'].snippet or "")[:200],
                        'Tags': ', '.join(r['item'].tags),
                        'Match_Score': f"{r['similarity']:.3f}"
                    } for r in results])
                    
                    st.download_button(
                        "📥 Export Results",
                        data=results_df.to_csv(index=False),
                        file_name=f"unl_geoportal_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # Display results
                for i, result in enumerate(results):
                    display_search_result(result, i)
                    
                # AI Chat about results
                if has_openai and user_message:
                    with st.expander("🤖 AI Assistant Response"):
                        ai_response = chat_with_user(user_message, results)
                        st.write(ai_response)
                        
            else:
                st.warning("No results found. Try different keywords or check the debug info in the sidebar.")
    
    # Example searches
    st.header("💡 Try These Searches")
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        if st.button("🏛️ census population data"):
            st.experimental_set_query_params(q="census population data")
            st.experimental_rerun()
    
    with example_col2:
        if st.button("🌾 agricultural land use"):
            st.experimental_set_query_params(q="agricultural land use")
            st.experimental_rerun()
    
    with example_col3:
        if st.button("🗻 elevation topography"):
            st.experimental_set_query_params(q="elevation topography")
            st.experimental_rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🏛️ <a href='https://geoportal.unl.edu/portal/apps/sites/#/unl-geoportal' target='_blank'>UNL Geoportal</a> | 
        🤖 Powered by AI & Semantic Search | 
        📚 <a href='https://libraries.unl.edu/' target='_blank'>UNL Libraries</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
