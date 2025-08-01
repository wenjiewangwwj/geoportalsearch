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

# Configure page
st.set_page_config(
    page_title="UNL Geoportal API Diagnostic",
    page_icon="🔧",
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

def test_api_endpoint(url: str, timeout: int = 10):
    """Test a single API endpoint and return diagnostic info"""
    try:
        st.write(f"🔍 Testing: `{url}`")
        
        response = requests.get(url, timeout=timeout)
        
        st.write(f"**Status Code:** {response.status_code}")
        st.write(f"**Content-Type:** {response.headers.get('content-type', 'Unknown')}")
        st.write(f"**Response Size:** {len(response.content)} bytes")
        
        if response.status_code == 200:
            try:
                data = response.json()
                st.write(f"**JSON Keys:** {list(data.keys()) if isinstance(data, dict) else type(data).__name__}")
                
                # Show first few characters of response
                text_preview = response.text[:500]
                with st.expander("Response Preview"):
                    st.code(text_preview, language="json")
                
                return True, data
            except json.JSONDecodeError:
                st.write("**Content:** Not JSON format")
                with st.expander("Raw Response"):
                    st.text(response.text[:1000])
                return False, None
        else:
            st.write(f"**Error:** {response.status_code} - {response.reason}")
            if response.text:
                with st.expander("Error Response"):
                    st.text(response.text[:500])
            return False, None
            
    except requests.exceptions.Timeout:
        st.error(f"⏰ Timeout after {timeout} seconds")
        return False, None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Connection failed")
        return False, None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return False, None

def test_search_endpoint(base_url: str, params: dict):
    """Test search functionality on an endpoint"""
    try:
        st.write(f"🔍 Testing search: `{base_url}` with params: {params}")
        
        response = requests.get(base_url, params=params, timeout=15)
        
        st.write(f"**Full URL:** {response.url}")
        st.write(f"**Status Code:** {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                
                # Look for common result fields
                total_results = 0
                results = []
                
                if 'results' in data:
                    results = data['results']
                    total_results = data.get('total', len(results))
                elif 'items' in data:
                    results = data['items']
                    total_results = len(results)
                elif isinstance(data, list):
                    results = data
                    total_results = len(results)
                
                st.write(f"**Total Results:** {total_results}")
                st.write(f"**Results Found:** {len(results)}")
                
                if results:
                    st.write("**Sample Result Keys:**")
                    sample = results[0]
                    st.write(list(sample.keys()) if isinstance(sample, dict) else type(sample).__name__)
                    
                    with st.expander("First Result"):
                        st.json(sample)
                
                return True, data
                
            except json.JSONDecodeError:
                st.error("Response is not valid JSON")
                return False, None
        else:
            st.error(f"Request failed: {response.status_code}")
            return False, None
            
    except Exception as e:
        st.error(f"Search test failed: {str(e)}")
        return False, None

def main():
    """Main diagnostic application"""
    
    st.title("🔧 UNL Geoportal API Diagnostic Tool")
    st.markdown("This tool helps diagnose API connectivity issues with the UNL Geoportal")
    
    # Test different base URLs
    st.header("🌐 Base URL Testing")
    
    base_urls = [
        "https://geoportal.unl.edu",
        "https://geoportal.unl.edu/portal",
        "https://geoportal.unl.edu/arcgis",
        "https://geoportal.unl.edu/server"
    ]
    
    working_bases = []
    
    for base_url in base_urls:
        with st.expander(f"Testing: {base_url}"):
            success, data = test_api_endpoint(base_url)
            if success:
                working_bases.append(base_url)
    
    # Test specific API endpoints
    st.header("🔍 API Endpoint Testing")
    
    potential_endpoints = [
        "https://geoportal.unl.edu/portal/sharing/rest",
        "https://geoportal.unl.edu/portal/sharing/rest/portals/info",
        "https://geoportal.unl.edu/portal/sharing/rest/search",
        "https://geoportal.unl.edu/arcgis/sharing/rest",
        "https://geoportal.unl.edu/arcgis/sharing/rest/search",
        "https://geoportal.unl.edu/server/rest/services",
        "https://geoportal.unl.edu/rest/services",
    ]
    
    working_endpoints = []
    
    for endpoint in potential_endpoints:
        with st.expander(f"Testing: {endpoint}"):
            success, data = test_api_endpoint(endpoint)
            if success:
                working_endpoints.append(endpoint)
    
    # Test search functionality
    st.header("🔎 Search Endpoint Testing")
    
    search_endpoints = [
        "https://geoportal.unl.edu/portal/sharing/rest/search",
        "https://geoportal.unl.edu/arcgis/sharing/rest/search",
        "https://geoportal.unl.edu/portal/sharing/rest/content/items",
    ]
    
    search_params = [
        {'q': '*', 'f': 'json', 'num': 5},
        {'q': 'census', 'f': 'json', 'num': 5},
        {'q': 'type:Feature Service', 'f': 'json', 'num': 5},
        {'f': 'json', 'num': 5}  # No query
    ]
    
    for endpoint in search_endpoints:
        st.subheader(f"Testing: {endpoint}")
        for i, params in enumerate(search_params):
            with st.expander(f"Search Test {i+1}: {params}"):
                success, data = test_search_endpoint(endpoint, params)
    
    # Manual endpoint testing
    st.header("🧪 Manual Endpoint Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        custom_url = st.text_input(
            "Custom URL to test:",
            value="https://geoportal.unl.edu/portal/sharing/rest/search"
        )
    
    with col2:
        test_params = st.text_area(
            "Parameters (JSON format):",
            value='{"q": "*", "f": "json", "num": 10}',
            height=100
        )
    
    if st.button("🚀 Test Custom Endpoint"):
        try:
            params = json.loads(test_params)
            with st.spinner("Testing..."):
                success, data = test_search_endpoint(custom_url, params)
        except json.JSONDecodeError:
            st.error("Invalid JSON in parameters")
    
    # Summary and recommendations
    st.header("📋 Summary & Recommendations")
    
    if working_endpoints:
        st.success(f"✅ Working endpoints found: {len(working_endpoints)}")
        for endpoint in working_endpoints:
            st.write(f"- {endpoint}")
    else:
        st.error("❌ No working API endpoints found")
        
        st.warning("**Possible Issues:**")
        st.write("1. **CORS Policy**: The server might block requests from external domains")
        st.write("2. **Authentication Required**: The API might require authentication")
        st.write("3. **Different API Structure**: UNL might use a custom API structure")
        st.write("4. **Network/Firewall**: Connection might be blocked")
        
    # Alternative approaches
    st.header("🔄 Alternative Approaches")
    
    st.write("**If API access fails, consider:**")
    st.write("1. **Web Scraping**: Parse the HTML pages directly")
    st.write("2. **Sitemap Parsing**: Look for XML sitemaps")
    st.write("3. **Manual Data Export**: Export data from the portal interface")
    st.write("4. **Contact UNL**: Ask for API documentation or access")
    
    # Test web scraping approach
    st.subheader("🕷️ Web Scraping Test")
    
    if st.button("Test Web Scraping Approach"):
        test_url = "https://geoportal.unl.edu/portal/apps/sites/#/unl-geoportal"
        
        try:
            response = requests.get(test_url, timeout=10)
            st.write(f"**Status:** {response.status_code}")
            st.write(f"**Content Length:** {len(response.content)} bytes")
            
            # Look for data indicators in the HTML
            content = response.text.lower()
            indicators = ['json', 'dataset', 'arcgis', 'feature', 'service']
            
            found_indicators = [ind for ind in indicators if ind in content]
            st.write(f"**Data Indicators Found:** {found_indicators}")
            
            # Show snippet of content
            with st.expander("HTML Content Preview"):
                st.code(response.text[:1000], language="html")
                
        except Exception as e:
            st.error(f"Web scraping test failed: {e}")

if __name__ == "__main__":
    main()