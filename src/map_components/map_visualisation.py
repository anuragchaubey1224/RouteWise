# map_visualisation.py

import folium
import pandas as pd
import os

def visualize_locations_on_map(
    df: pd.DataFrame,
    latitude_col: str,
    longitude_col: str,
    popup_cols: list = None,
    zoom_start: int = 12,
    map_title: str = "Location Visualization",
    output_html_path: str = "/Users/anuragchaubey/RouteWise/outputs/map_output.html"
):
    """
    visualizes individual locations on a Folium map.
    """
    if df.empty:
        print("⚠️ dataframe empty")
        return

    if latitude_col not in df.columns or longitude_col not in df.columns:
        print(f" latitude  '{latitude_col}' or '{longitude_col}' columns are missing in the DataFrame")
        return

    # define the map center based on the first non-null location
    center_lat = df[latitude_col].dropna().iloc[0]
    center_lon = df[longitude_col].dropna().iloc[0]
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, control_scale=True)

    # add markers for each location
    for _, row in df.iterrows():
        lat, lon = row[latitude_col], row[longitude_col]
        if pd.isna(lat) or pd.isna(lon):
            continue

        popup_text = f"<b>Lat:</b> {lat}<br><b>Lon:</b> {lon}"
        if popup_cols:
            popup_text += ''.join([
                f"<br><b>{col}:</b> {row[col]}" for col in popup_cols
                if col in row and not pd.isna(row[col])
            ])

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip="click for details",
        ).add_to(m)

    # title added to the map
    m.get_root().html.add_child(folium.Element(
        f'<h3 align="center" style="font-size:20px"><b>{map_title}</b></h3>'
    ))
    # map saved in HTML file
    m.save(output_html_path)
    print(f"map saved: {output_html_path}")


def visualize_delivery_routes_on_map(
    df: pd.DataFrame,
    pickup_lat_col: str,
    pickup_lon_col: str,
    delivery_lat_col: str,
    delivery_lon_col: str,
    popup_cols: list = None,
    zoom_start: int = 12,
    map_title: str = "Delivery Route Visualization",
    output_html_path: str = "/Users/anuragchaubey/RouteWise/outputs/delivery_routes_map.html"
):
    """
    visualizes delivery routes (pickup → delivery) on a Folium map with predicted time and order info
    """
    if df.empty:
        print(" dataframe empty")
        return

    required_cols = [pickup_lat_col, pickup_lon_col, delivery_lat_col, delivery_lon_col]
    if not all(col in df.columns for col in required_cols):
        print(f"missing required column: {', '.join(required_cols)}")
        return

    # define the map center based on the average of pickup and delivery locations
    center_lat = pd.concat([df[pickup_lat_col], df[delivery_lat_col]]).dropna().mean()
    center_lon = pd.concat([df[pickup_lon_col], df[delivery_lon_col]]).dropna().mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, control_scale=True)

    # add markers and lines for each route
    for _, row in df.iterrows():
        p_lat, p_lon = row[pickup_lat_col], row[pickup_lon_col]
        d_lat, d_lon = row[delivery_lat_col], row[delivery_lon_col]

        if pd.isna(p_lat) or pd.isna(p_lon) or pd.isna(d_lat) or pd.isna(d_lon):
            continue

        popup_text = ""
        if popup_cols:
            popup_text += ''.join([
                f"<b>{col}:</b> {row[col]}<br>" for col in popup_cols
                if col in row and not pd.isna(row[col])
            ])
        popup_text += f"<b>pickup:</b> ({p_lat:.4f}, {p_lon:.4f})<br><b>delivery:</b> ({d_lat:.4f}, {d_lon:.4f})"

        # pickup marker
        folium.Marker(
            location=[p_lat, p_lon],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color='green', icon='truck', prefix='fa'),
            tooltip="pickup position"
        ).add_to(m)

        # delivery marker
        folium.Marker(
            location=[d_lat, d_lon],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color='red', icon='info-sign'),
            tooltip="delivery position"
        ).add_to(m)

        # route line
        folium.PolyLine(
            locations=[[p_lat, p_lon], [d_lat, d_lon]],
            color='blue',
            weight=2.5,
            opacity=0.7,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(m)

    # add title to the map
    m.get_root().html.add_child(folium.Element(
        f'<h3 align="center" style="font-size:20px"><b>{map_title}</b></h3>'
    ))
    # map saved in HTML file
    m.save(output_html_path)
    print(f" map saved: {output_html_path} ")


# local testing (optional)
if __name__ == "__main__":
    print("map_visualisation.py - local testing")
    output_base_dir = "/Users/anuragchaubey/RouteWise/outputs"
    os.makedirs(output_base_dir, exist_ok=True)

    sample_df = pd.DataFrame({
        'order_id': [1, 2, 3],
        'pickup_lat': [12.9716, 13.0827, 12.9165],
        'pickup_lon': [77.5946, 80.2707, 77.6000],
        'delivery_lat': [12.9279, 13.0679, 12.9200],
        'delivery_lon': [77.6271, 80.2185, 77.5800],
        'predicted_time': [30, 45, 60]
    })

    test_output_path = os.path.join(output_base_dir, "sample_delivery_routes_map.html")
    visualize_delivery_routes_on_map(
        df=sample_df,
        pickup_lat_col='pickup_lat',
        pickup_lon_col='pickup_lon',
        delivery_lat_col='delivery_lat',
        delivery_lon_col='delivery_lon',
        popup_cols=['order_id', 'predicted_time'],
        map_title="Sample Delivery Routes",
        output_html_path=test_output_path
    )
    print(f"sample map saved: {test_output_path}")
