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
from sentence_transformers import SentenceTransformer
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
    thumbnail: Optional[str] = None
    extent: Optional[Dict] = None
    num_views: Optional[int] = None
    snippet: Optional[str] = None

class ArcGISPortalConnector:
    """Handles connection and data retrieval from ArcGIS Portal"""
    
    def __init__(self, portal_url: str):
        self.portal_url = portal_url.rstrip('/')
        self.session = requests.Session()
        # Set a reasonable timeout
        self.session.timeout = 30
    
    def search_content(self, query: str = "*", num: int = 100, start: int = 1) -> Dict:
        """Search for content in the portal"""
        search_url = f"{self.portal_url}/search"
        
        params = {
            'q': query,
            'num': min(num, 100),  # Portal usually limits to 100
            'start': start,
            'f': 'json',
            'sortField': 'modified',
            'sortOrder': 'desc'
        }
        
        try:
            response = self.session.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Search request failed: {e}")
            return {'results': [], 'total': 0}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return {'results': [], 'total': 0}
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            return {'results': [], 'total': 0}
    
    def get_all_content(self, max_items: int = 1000) -> List[Dict]:
        """Get all available content from the portal"""
        all_items = []
        start = 1
        num_per_request = 100
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while len(all_items) < max_items:
            status_text.text(f"Loading data... {len(all_items)} items retrieved")
            
            result = self.search_content(query="*", num=num_per_request, start=start)
            
            if 'results' not in result or not result['results']:
                break
                
            all_items.extend(result['results'])
            
            # Update progress
            if 'total' in result and result['total'] > 0:
                progress = min(len(all_items) / min(result['total'], max_items), 1.0)
                progress_bar.progress(progress)
            
            # Check if we've got all available items
            if len(result['results']) < num_per_request:
                break
                
            start += num_per_request
            
            # Avoid overwhelming the server
            time.sleep(0.1)
        
        progress_bar.progress(1.0)
        status_text.text(f"Loaded {len(all_items)} items successfully!")
        
        return all_items[:max_items]

@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model (cached)"""
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_portal_data():
    """Load and process data from the portal (cached)"""
    portal = ArcGISPortalConnector("https://geoportal.unl.edu/portal/sharing/rest")
    
    with st.spinner("Loading data from UNL Geoportal..."):
        raw_items = portal.get_all_content(max_items=2000)
    
    # Convert to GISDataItem objects
    data_items = []
    for item in raw_items:
        try:
            # Create the portal URL for the item
            item_url = f"https://geoportal.unl.edu/portal/home/item.html?id={item.get('id')}"
            
            gis_item = GISDataItem(
                id=item.get('id', ''),
                title=item.get('title', 'Untitled'),
                description=item.get('description', '') or item.get('snippet', ''),
                tags=item.get('tags', []),
                owner=item.get('owner', 'Unknown'),
                created=item.get('created', ''),
                modified=item.get('modified', ''),
                type=item.get('type', 'Unknown'),
                url=item_url,
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
def create_embeddings(data_items):
    """Create embeddings for all data items (cached)"""
    model = load_embedding_model()
    if not model:
        return None, None
    
    # Prepare text for embedding
    texts = []
    for item in data_items:
        # Create rich text representation
        text_parts = [
            item.title,
            item.description or item.snippet or "",
            " ".join(item.tags),
            item.type
        ]
        text = ". ".join([part for part in text_parts if part.strip()])
        texts.append(text)
    
    # Create embeddings
    with st.spinner("Creating search embeddings..."):
        embeddings = model.encode(texts, show_progress_bar=True)
    
    return embeddings, texts

def semantic_search(query: str, data_items: List[GISDataItem], embeddings, texts, top_k: int = 10):
    """Perform semantic search using embeddings"""
    model = load_embedding_model()
    if not model or embeddings is None:
        return []
    
    # Create query embedding
    query_embedding = model.encode([query])
    
    # Calculate similarities
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    
    # Get top results
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:  # Minimum similarity threshold
            item = data_items[idx]
            results.append({
                'item': item,
                'similarity': float(similarities[idx]),
                'text': texts[idx]
            })
    
    return results

def keyword_search(query: str, data_items: List[GISDataItem], top_k: int = 10):
    """Fallback keyword search"""
    query_lower = query.lower()
    query_words = re.findall(r'\w+', query_lower)
    
    results = []
    for item in data_items:
        score = 0
        text_to_search = f"{item.title} {item.description} {' '.join(item.tags)} {item.type}".lower()
        
        for word in query_words:
            if word in text_to_search:
                score += text_to_search.count(word)
        
        if score > 0:
            results.append({
                'item': item,
                'similarity': score / len(query_words),
                'text': text_to_search[:200]
            })
    
    # Sort by score
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_k]

def display_search_result(result, index):
    """Display a single search result"""
    item = result['item']
    similarity = result['similarity']
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Title with link
            st.markdown(f"### [{item.title}]({item.url})")
            
            # Type badge
            st.markdown(f"**Type:** `{item.type}`")
            
            # Description
            if item.description:
                st.write(item.description[:300] + "..." if len(item.description) > 300 else item.description)
            elif item.snippet:
                st.write(item.snippet[:300] + "..." if len(item.snippet) > 300 else item.snippet)
            
            # Tags
            if item.tags:
                tags_html = " ".join([f'<span style="background-color: #e6f3ff; padding: 2px 6px; border-radius: 3px; font-size: 12px; margin: 2px;">{tag}</span>' for tag in item.tags[:8]])
                st.markdown(f"**Tags:** {tags_html}", unsafe_allow_html=True)
            
            # Metadata
            col_meta1, col_meta2, col_meta3 = st.columns(3)
            with col_meta1:
                st.caption(f"👤 Owner: {item.owner}")
            with col_meta2:
                if item.num_views:
                    st.caption(f"👁️  Views: {item.num_views}")
            with col_meta3:
                st.caption(f"📅 Modified: {item.modified[:10] if item.modified else 'Unknown'}")
        
        with col2:
            # Similarity score
            st.metric("Match Score", f"{similarity:.1%}")
            
            # Action buttons
            st.markdown(f"[🔗 View in Portal]({item.url})")
            if item.thumbnail:
                thumbnail_url = f"https://geoportal.unl.edu/portal/sharing/rest/content/items/{item.id}/info/{item.thumbnail}"
                st.markdown(f"[🖼️ Thumbnail]({thumbnail_url})")
        
        st.divider()

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🗺️ UNL Geoportal Intelligent Search")
    st.markdown("Search for geospatial data using natural language. No need for technical keywords!")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This tool helps you find geospatial data from the UNL Geoportal using natural language queries.
        
        **Examples:**
        - "population data for Nebraska counties"
        - "agricultural land use maps"
        - "elevation and topography data"
        - "water resources and rivers"
        - "transportation networks"
        """)
        
        st.header("🔧 Settings")
        search_method = st.selectbox(
            "Search Method",
            ["Semantic Search (AI)", "Keyword Search"],
            help="Semantic search uses AI to understand meaning, while keyword search looks for exact matches."
        )
        
        num_results = st.slider("Number of Results", 5, 50, 10)
        
        st.header("📊 Portal Stats")
        
    # Load data
    try:
        data_items = load_portal_data()
        st.sidebar.success(f"✅ {len(data_items)} datasets loaded")
        
        # Create embeddings for semantic search
        embeddings, texts = None, None
        if search_method == "Semantic Search (AI)":
            embeddings, texts = create_embeddings(data_items)
            if embeddings is not None:
                st.sidebar.success("✅ AI search ready")
            else:
                st.sidebar.error("❌ AI search unavailable")
                search_method = "Keyword Search"
        
    except Exception as e:
        st.error(f"Failed to load portal data: {e}")
        st.stop()
    
    # Search interface
    st.header("🔍 Search")
    
    # Search input
    query = st.text_input(
        "What geospatial data are you looking for?",
        placeholder="e.g., population demographics, crop yield data, flood maps...",
        help="Describe what you're looking for in plain English"
    )
    
    # Search button
    if st.button("Search", type="primary") or query:
        if query.strip():
            with st.spinner("Searching..."):
                # Perform search
                if search_method == "Semantic Search (AI)" and embeddings is not None:
                    results = semantic_search(query, data_items, embeddings, texts, num_results)
                else:
                    results = keyword_search(query, data_items, num_results)
                
                # Display results
                if results:
                    st.success(f"Found {len(results)} relevant datasets")
                    
                    # Add export options
                    if len(results) > 0:
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            # Create downloadable results
                            results_df = pd.DataFrame([{
                                'Title': r['item'].title,
                                'Type': r['item'].type,
                                'Owner': r['item'].owner,
                                'URL': r['item'].url,
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
                        
                else:
                    st.warning("No results found. Try different keywords or check the sidebar for search tips.")
        else:
            st.warning("Please enter a search query.")
    
    # Example queries
    st.header("💡 Example Searches")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌾 Agricultural Data"):
            st.rerun()
    
    with col2:
        if st.button("🏙️ Urban Planning"):
            st.rerun()
    
    with col3:
        if st.button("💧 Water Resources"):
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🏛️ <a href='https://geoportal.unl.edu/portal/apps/sites/#/unl-geoportal' target='_blank'>UNL Geoportal</a> | 
        🔬 Powered by AI Semantic Search | 
        📚 <a href='https://libraries.unl.edu/' target='_blank'>UNL Libraries</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
