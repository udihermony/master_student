import urllib.request
import json

def run(city: str) -> str:
    """Get current weather for a given city."""
    try:
        url = f"https://wttr.in/{city}?format=j1"
        req = urllib.request.Request(url, headers={"User-Agent": "curl/7.68.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
        
        current = data["current_condition"][0]
        temp_c = current["temp_C"]
        temp_f = current["temp_F"]
        desc = current["weatherDesc"][0]["value"]
        humidity = current["humidity"]
        wind_kmph = current["windspeedKmph"]
        feels_like_c = current["FeelsLikeC"]
        
        return json.dumps({
            "city": city,
            "temperature_c": temp_c,
            "temperature_f": temp_f,
            "feels_like_c": feels_like_c,
            "conditions": desc,
            "humidity_percent": humidity,
            "wind_kmph": wind_kmph
        })
    except Exception as e:
        return json.dumps({"error": str(e), "city": city})
