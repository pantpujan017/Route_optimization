import streamlit as st
import heapq
import pandas as pd
import folium
from folium import plugins
import json
import geopandas as gpd
from shapely.geometry import Point, LineString
import osmnx as ox
import requests

city_coordinates = {
    'Baglung': [28.2722, 83.5977],
    'Bhaktapur': [27.6710, 85.4298],
    'Chitwan': [27.5291, 84.3542],
    'Dhading': [27.9711, 84.8985],
    'Dolakha': [27.7784, 86.1752],
    'Gorkha': [28.0000, 84.6333],
    'Kaski': [28.2622, 84.0167],
    'Kathmandu': [27.7172, 85.3240],
    'Kavrepalanchowk': [27.5285, 85.5620],
    'Lalitpur': [27.6588, 85.3247],
    'Lamjung': [28.2765, 84.3542],
    'Makwanpur': [27.4323, 85.0316],
    'Manang': [28.6667, 84.0167],
    'Mustang': [29.0000, 83.8333],
    'Myagdi': [28.6000, 83.3333],
    'Nawalpur': [27.7833, 84.1167],
    'Nuwakot': [27.9167, 85.1667],
    'Parbat': [28.2333, 83.6833],
    'Ramechhap': [27.3333, 86.0833],
    'Rasuwa': [28.1667, 85.3333],
    'Sindhuli': [27.2569, 85.9713],
    'Sindhupalchowk': [27.9500, 85.6833],
    'Syangja': [28.0833, 83.8667],
    'Tanahu': [27.9447, 84.2279]
}




graph_distances = {
    'Baglung': {'Myagdi': 29.0, 'Parbat': 23.0},
    'Bhaktapur': {'Kathmandu': 15.0, 'Lalitpur': 12.0, 'Kavrepalanchowk': 42.0, 'Sindhupalchowk': 67.0},
    'Chitwan': {'Nawalpur': 34.4, 'Makwanpur': 103.0, 'Dhading': 121.0, 'Tanahu': 113.0, 'Gorkha': 69.0},
    'Dhading': {'Kathmandu': 112.0, 'Gorkha': 67.0, 'Nuwakot': 66.6, 'Makwanpur': 152.0, 'Chitwan': 121.0},
    'Dolakha': {'Ramechhap': 33.0, 'Kavrepalanchowk': 112.0},
    'Gorkha': {'Chitwan': 69.0, 'Dhading': 67.0, 'Tanahu': 58.0, 'Lamjung': 81.0},
    'Kaski': {'Parbat': 56.0, 'Syangja': 36.8, 'Tanahu': 50.1},
    'Kathmandu': {'Lalitpur': 7.5, 'Bhaktapur': 15.0, 'Nuwakot': 59.0, 'Dhading': 112.0, 'Makwanpur': 88.0,
                  'Sindhupalchowk': 68.0},
    'Kavrepalanchowk': {'Bhaktapur': 42.0, 'Sindhupalchowk': 68.4, 'Dolakha': 112.0},
    'Lalitpur': {'Kathmandu': 7.5, 'Bhaktapur': 12.0},
    'Lamjung': {'Gorkha': 81.0, 'Tanahu': 60.0, 'Manang': 67.7},
    'Makwanpur': {'Chitwan': 103.0, 'Dhading': 152.0, 'Kathmandu': 88.0, 'Sindhuli': 125.0},
    'Manang': {'Lamjung': 67.7},
    'Mustang': {'Myagdi': 85.4},
    'Myagdi': {'Baglung': 29.0, 'Parbat': 42.2, 'Mustang': 85.4},
    'Nawalpur': {'Chitwan': 34.4},
    'Nuwakot': {'Dhading': 66.6, 'Rasuwa': 49.4, 'Sindhupalchowk': 98.4, 'Kathmandu': 59.0},
    'Parbat': {'Baglung': 23.0, 'Myagdi': 42.2, 'Kaski': 56.0, 'Syangja': 54.5},
    'Ramechhap': {'Sindhuli': 68.1, 'Dolakha': 33.0},
    'Rasuwa': {'Nuwakot': 49.4},
    'Sindhuli': {'Makwanpur': 125.0, 'Ramechhap': 68.1},
    'Sindhupalchowk': {'Bhaktapur': 67.0, 'Kavrepalanchowk': 68.4, 'Nuwakot': 98.4},
    'Syangja': {'Parbat': 54.5, 'Kaski': 36.8},
    'Tanahu': {'Kaski': 50.1, 'Lamjung': 60.0, 'Gorkha': 58.0, 'Chitwan': 113.0}
}


def dijkstra(graph, start_city):
    """Run Dijkstra's algorithm to find the shortest path from start_city to all other cities."""
    # Initially, all distances are set to infinity except start_city = 0

    distances = {city: float('inf') for city in graph}
    distances[start_city] = 0
    priority_queue = [(0, start_city)]  # (distance, city)
    # current_city is the city we're currently exploring, and
    # current_distance is the shortest known distance to that city.
    while priority_queue:
        current_distance, current_city = heapq.heappop(priority_queue)

        for neighbor, weight in graph[current_city].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                # heap will automatically adjust to keep the smallest element
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances


def create_graph(graph):
    """Calculate and add heuristic values h(n) for each city using the shortest path distances."""
    heuristic_graph = {}
    for city in graph:
        # Get shortest paths from the current city to all other cities
        shortest_paths = dijkstra(graph, city)
        # **graph[city] unpacks the original city's distance information, and
        # 'h': shortest_paths adds the heuristic values
        heuristic_graph[city] = {**graph[city], 'h': shortest_paths}  # Add h(n)

    return heuristic_graph


def find_all_paths(graph, start_city, goal_city):
    """Find all possible paths from start_city to goal_city."""
    open_list = [(start_city, [start_city], 0)]  # (current city, path so far, current cost)
    all_paths = []  # To store all possible paths along with their costs

    while open_list:
        current_city, path, current_cost = open_list.pop()

        # If we've reached the goal, add the path to the list
        if current_city == goal_city:
            all_paths.append((path, current_cost))
            continue

        # Explore neighbors
        for neighbor, g_cost in graph[current_city].items():
            if neighbor == 'h':  # Skip heuristic
                continue

            # If the neighbor is not already in the path (to avoid cycles), explore further
            if neighbor not in path:
                new_path = path + [neighbor]
                new_cost = current_cost + g_cost
                open_list.append((neighbor, new_path, new_cost))

    return all_paths


def a_star_for_best_path(graph, path):
    """Apply A* to evaluate the quality of a given path."""
    total_cost = 0
    total_length = 0
    for i in range(len(path) - 1):
        current_city = path[i]
        next_city = path[i + 1]
        total_cost += graph[current_city][next_city]  # g(n)
        total_length += 1  # Path length

    # Custom "best" criteria: Consider total cost, path length, and heuristic value
    heuristic_cost = graph[path[-1]]['h'][path[-1]]  # Heuristic from the last node to itself (0 for simplicity)
    f_cost = total_cost + heuristic_cost

    return f_cost, total_length


def get_best_path_based_on_criteria(all_paths, graph):
    """Find the best path based on the custom criteria."""
    best_path = None
    best_score = float('inf')  # Use this to track the best score

    for path, cost in all_paths:
        f_cost, total_length = a_star_for_best_path(graph, path)

        # Apply the custom criteria: Here we can adjust it based on the needs (minimize f_cost or length, etc.)
        score = f_cost  # Or some weighted combination of f_cost and total_length

        if score < best_score:
            best_score = score
            best_path = path

    return best_path, best_score


import osmnx as ox
import folium


import osmnx as ox
import folium


import geopandas as gpd
from shapely.geometry import Point, LineString
import folium


def create_route_map(path, city_coordinates):
    """Create a Folium map showing the route between cities using OSRM API."""

    # Get coordinates for the path
    path_coordinates = [city_coordinates[city] for city in path]

    # Calculate center of the route (average coordinates)
    center_lat = sum(coord[0] for coord in path_coordinates) / len(path_coordinates)
    center_lng = sum(coord[1] for coord in path_coordinates) / len(path_coordinates)

    # Create the map
    m = folium.Map(location=[center_lat, center_lng], zoom_start=8)

    # Add markers for each city
    for city, coords in zip(path, path_coordinates):
        folium.Marker(
            coords,
            popup=city,
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)

    # OSRM API base URL for routing
    osrm_url = "http://router.project-osrm.org/route/v1/driving"

    # Create the route using OSRM (road routes, not straight lines)
    route_coordinates = []
    for i in range(len(path_coordinates) - 1):
        # Define the start and end coordinates for each segment
        start = path_coordinates[i]
        end = path_coordinates[i + 1]

        # Query OSRM API for driving directions between two cities
        route_url = f"{osrm_url}/{start[1]},{start[0]};{end[1]},{end[0]}?overview=full&geometries=geojson"
        response = requests.get(route_url)

        # Extract the route geometry (coordinates)
        data = response.json()
        if 'routes' in data and len(data['routes']) > 0:
            geometry = data['routes'][0]['geometry']['coordinates']
            route_coordinates.extend([(lat, lon) for lon, lat in geometry])

    # Add the route to the map
    folium.PolyLine(
        route_coordinates,
        weight=3,
        color='blue',
        opacity=0.8
    ).add_to(m)

    return m




df = pd.read_csv("cleanPlaceWithDistanceheadquater.csv")


def recommend_places(df, district_name, tags_list):
    """Filter and recommend places based on district and tags."""
    filtered_df = df[df["district"] == district_name]

    def has_tags(row_tags):
        place_tags = set(map(str.strip, row_tags.lower().split(",")))
        return any(tag.lower() in place_tags for tag in tags_list)

    recommended_places = filtered_df[filtered_df["tags"].apply(has_tags)]

    recommendations = []
    for _, row in recommended_places.iterrows():
        filtered_tags = [tag for tag in row['tags'].split(",") if tag.strip().lower() in tags_list]
        recommendations.append({
            'pID': row['pID'],
            'pName': row['pName'],
            'tags': ', '.join(filtered_tags),
            'district': row['district'],
            'headquarter': row['Headquarters'],
            'distance_km': row['distance_km']
        })
    return recommendations


def display_recommendations(path, tags_list):
    """Display recommendations for each city in a path based on selected tags, with improved formatting."""
    for city in path:
        recommendations = recommend_places(df, city, tags_list)
        if recommendations:
            st.markdown(f"### 🌄 _Places to Visit in **{city}**_")
            # Instead of using an expander, create a container with styled recommendations
            for place in recommendations[:2]:  # Show only the top 2 recommendations per city
                st.markdown(f"""
                <div style='background-color: #1E90FF; padding: 10px; border-radius: 8px; margin: 5px 0;'>
                    <h4 style='color: #1E4B6C; margin: 0;'>📍 {place['pName']}</h4>
                    <p style='margin: 5px 0;'><strong>🏷️ Tags:</strong> {place['tags']}</p>
                    <p style='margin: 5px 0;'><strong>📏 Distance:</strong> {place['distance_km']} km from {place['headquarter']}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("---")

# st.set_page_config(page_title="Nepal Travel Path Finder", layout="wide")
# st.title("Nepal Travel Path Finder 🚗")
# st.write("Explore the best path between cities and find recommended places along the way!")

# Tag selection
available_tags = ['heritage', 'temple', 'religious shrine', 'statue', 'sightseeing', 'historical',
                  'river', 'lake', 'wildlife', 'trekking', 'religious', 'mountain', 'hill', 'cultural',
                  'village', 'sculpture', 'museum', 'hiking', 'picnic spot', 'boat tour', 'pond',
                  'cycling', 'red panda', 'monastery', 'hindu', 'buddhist', 'scenary',
                  'rafting', 'scenery', 'rhododendron', 'dam', 'club', 'scultptures', 'tour',
                  'casino', 'street', 'garden', 'waterfall', 'farm', 'biking', 'dam-site',
                  'monsatery', 'fort', 'adventure', 'natural beauty', 'stupa',
                  'border', 'zoo', 'history', 'palace', 'art', 'landmark', 'cable car',
                  'park', 'cave', 'climbing', 'city', 'gumba', 'national park', 'pilgrimage',
                  'forest', 'homestay', 'paragliding', 'rappelling', 'canyoning', 'bungee']

cities = list(graph_distances.keys())
# start_city = st.selectbox("Select your starting city:", cities)
# goal_city = st.selectbox("Select your goal city:", cities)
# tags_list = st.multiselect("Select tags for place recommendations:", available_tags)

import streamlit as st
import heapq
import pandas as pd
from pathlib import Path

st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        min-height: 100vh;
        padding: 2rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #16222A 0%, #3A6073 100%);
        padding: 2rem 1rem;
    }

    /* Sidebar text color */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .element-container {
        color: #e0e0e0 !important;
    }

    /* Sidebar title */
    [data-testid="stSidebar"] h2 {
        color: #00ff9d;
        padding: 0.5rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid rgba(0, 255, 157, 0.2);
    }

    /* Sidebar widgets */
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stTextInput > div > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(0, 255, 157, 0.2);
        color: #e0e0e0 !important;
    }

    /* Custom container styling */
    .css-1d391kg {
        background-color: #1e1e1e;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 255, 157, 0.1);
    }

    /* Title and headers */
    h1 {
        color: #00ff9d;
        font-size: 2.2em;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 15px;
        background: linear-gradient(135deg, #16222A, #3A6073);
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 255, 157, 0.2);
    }

    h2, h3 {
        color: #00ff9d;
        margin-top: 1.5rem;
    }

    /* Card styling */
    .recommendation-card {
        background-color: #2a2a2a;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #2a2a2a;
        box-shadow: 0 2px 8px rgba(0, 255, 157, 0.1);
    }

    /* Path visualization */
    .city-path {
        font-size: 1.1em;
        color: #e0e0e0;
        padding: 10px;
        background-color: #2a2a2a;
        border-radius: 6px;
        margin: 5px 0;
        border: 1px solid #00ff9d;
    }

    /* Button styling */
    .stButton > button {
        background-color: #2a2a2a;
        color: #0a0a0a;
        border-radius: 6px;
        padding: 8px 20px;
        font-weight: 500;
        border: none;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #00cc7d;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 255, 157, 0.2);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #2a2a2a;
        border-radius: 6px;
        color: #00ff9d !important;
        padding: 10px;
    }

    /* Expander content */
    .streamlit-expanderContent {
        background-color: #1e1e1e;
        border-radius: 0 0 6px 6px;
        padding: 15px;
        border: 1px solid #00ff9d;
    }

    /* Tabs styling */
    .stTabs > div > div > div {
        background-color: #2a2a2a !important;
        color: #e0e0e0 !important;
        border-radius: 6px 6px 0 0;
    }

    .stTabs > div > div > div[aria-selected="true"] {
        background-color: #00ff9d !important;
        color: #0a0a0a !important;
    }

    /* Input fields */
    .stTextInput > div > div {
        background-color: #2a2a2a !important;
        border: 1px solid #00ff9d;
        color: #e0e0e0 !important;
    }

    /* Multiselect background */
    .stMultiSelect > div {
        background-color: #2a2a2a !important;
        border: 1px solid #00ff9d;
    }

    /* Text color override */
    p, li, label, span {
        color: #e0e0e0 !important;
    }

    /* Tables */
    .stTable {
        background-color: #1e1e1e;
        border-radius: 6px;
        overflow: hidden;
        border: 1px solid #00ff9d;
    }
</style>
""", unsafe_allow_html=True)



# Title section with orange background
st.markdown("""
    <h1>🏔️ Nepal Travel Path Finder</h1>
    <p style='text-align: center; font-size: 1.2em; color: #1a237e; background-color: #2a2a2a; padding: 10px; border-radius: 8px;'>
        Discover the perfect journey through Nepal's beautiful destinations
    </p>
""", unsafe_allow_html=True)

# Create two columns for the main input section
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🚩 Start Your Journey")
    start_city = st.selectbox("Starting City:", cities)

with col2:
    st.markdown("### 🎯 Your Destination")
    goal_city = st.selectbox("Goal City:", cities)

# Create an organized tag selection section
st.markdown("### 🏷️ Choose Your Interests")
st.markdown("Select the types of places you'd like to visit along your journey:")
tags_list = st.multiselect("", available_tags)

# Update the find paths button
if st.button("🔍 Find Best Travel Paths"):
    if start_city == goal_city:
        st.error("🚫 Start and Goal cities cannot be the same!")
    else:
        st.markdown("### 🔄 Planning Your Journey...")

        with st.spinner("Calculating optimal paths..."):
            progress_bar = st.progress(0)
            graph_with_heuristic = create_graph(graph_distances)
            all_paths = find_all_paths(graph_with_heuristic, start_city, goal_city)

            progress_bar.progress(50)

            if all_paths:
                top_paths = sorted(all_paths, key=lambda x: x[1])[:3]
                progress_bar.progress(80)

                # Create tabs for routes and recommendations
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "🗺️ All Routes",
                    "🏃 Route 1 Places",
                    "🏃 Route 2 Places",
                    "🏃 Route 3 Places",
                    "⭐ Best Route"
                ])

                # All Routes Overview Tab
                with tab1:
                    st.markdown("### Top 3 Routes Overview")
                    for idx, (path, cost) in enumerate(top_paths, 1):
                        with st.container():
                            st.markdown(f"""
                            <div class='recommendation-card'>
                                <h4>Route {idx}</h4>
                                <div class='city-path'>{' ➔ '.join(path)}</div>
                                <p>Total Distance: <strong>{cost:.1f} km</strong></p>
                            </div>
                            """, unsafe_allow_html=True)

                            # Create and display map for this route
                            route_map = create_route_map(path,city_coordinates)

                            # Save the map to a temporary HTML file
                            map_html = f'route_map_{idx}.html'
                            route_map.save(map_html)

                            # Display the map using an iframe
                            st.components.v1.html(
                                open(map_html, 'r').read(),
                                height=400
                            )
                            st.markdown("---")

                # Route 1 Recommendations Tab
                with tab2:
                    path1, cost1 = top_paths[0]
                    st.markdown("### 🎯 Route 1 Recommended Places")
                    st.markdown(f"""
                    <div class='recommendation-card'>
                        <h4>Route 1 Path</h4>
                        <div class='city-path'>{' ➔ '.join(path1)}</div>
                        <p>Total Distance: <strong>{cost1:.1f} km</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    display_recommendations(path1, tags_list)

                # Route 2 Recommendations Tab
                with tab3:
                    path2, cost2 = top_paths[1]
                    st.markdown("### 🎯 Route 2 Recommended Places")
                    st.markdown(f"""
                    <div class='recommendation-card'>
                        <h4>Route 2 Path</h4>
                        <div class='city-path'>{' ➔ '.join(path2)}</div>
                        <p>Total Distance: <strong>{cost2:.1f} km</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    display_recommendations(path2, tags_list)

                # Route 3 Recommendations Tab
                with tab4:
                    path3, cost3 = top_paths[2]
                    st.markdown("### 🎯 Route 3 Recommended Places")
                    st.markdown(f"""
                    <div class='recommendation-card'>
                        <h4>Route 3 Path</h4>
                        <div class='city-path'>{' ➔ '.join(path3)}</div>
                        <p>Total Distance: <strong>{cost3:.1f} km</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    display_recommendations(path3, tags_list)

                # Best Route Tab
                with tab5:
                    best_path, best_score = get_best_path_based_on_criteria(top_paths, graph_with_heuristic)
                    st.markdown("### 🏆 Optimal Travel Route")
                    st.markdown(f"""
                    <div class='recommendation-card'>
                        <h4>Best Path Based on Your Preferences</h4>
                        <div class='city-path'>{' ➔ '.join(best_path)}</div>
                        <p>Optimized Score: <strong>{best_score:.1f}</strong></p>
                        <div class="tag">Recommended ⭐</div>
                    </div>
                    """, unsafe_allow_html=True)
                    display_recommendations(best_path, tags_list)

                progress_bar.progress(100)

            else:
                st.error(f"❌ No path found from {start_city} to {goal_city}")
                progress_bar.progress(100)

# Sidebar with orange background
with st.sidebar:
    st.markdown("""
    # 🏔️ About Nepal Travel

    Discover the beauty of Nepal through our intelligent travel planning system.

    ### 📝 How to Use:
    1. Select your starting city
    2. Choose your destination
    3. Pick your interests from the tags
    4. Click 'Find Best Travel Paths'

    ### 🎯 Features:
    - Multiple route options
    - Place recommendations
    - Distance calculations
    - Optimized paths

    ### 🏞️ Popular Destinations:
    - Kathmandu Valley
    - Pokhara
    - Chitwan
    - And many more!
    """)