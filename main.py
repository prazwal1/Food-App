from flask import request, jsonify, Flask, render_template, send_from_directory, session
from flask_cors import CORS
import os
import tempfile
import pandas as pd
import io
from src import train_demand_model, predict_next_hour, predict_next_24_hours, generate_realistic_hourly_orders
import pathlib
import secrets

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# Enable CORS if needed
CORS(app)

# Session-based storage - each session gets its own model
# Structure: sessions = {session_id: {'model': model, 'df': df, 'feature_cols': [...], 'dish_cols': [...]}}
sessions = {}

if not os.path.exists("./model"):
    os.makedirs("./model")

if not os.path.exists("./static"):
    os.makedirs("./static")

def get_session_id():
    """Get or create a session ID for the current user"""
    if 'user_id' not in session:
        session['user_id'] = secrets.token_hex(16)
    return session['user_id']

def get_session_data():
    """Get session-specific data"""
    session_id = get_session_id()
    if session_id not in sessions:
        sessions[session_id] = {
            'model': None,
            'df': None,
            'feature_cols': None,
            'dish_cols': None
        }
    return sessions[session_id]

@app.route("/health", methods=["GET"])
def health():
    session_data = get_session_data()
    return jsonify({
        "status": "ok",
        "model_loaded": session_data['model'] is not None,
        "session_id": get_session_id()
    })

@app.route("/generate", methods=["POST"])
def generate():
    """
    Generate synthetic dataset for the current session.
    Does not train the model - just generates and returns the data.
    """
    data = request.get_json(silent=True) or {}
    rows = int(data.get("rows", 1008))
    
    try:
        df_gen = generate_realistic_hourly_orders(target_rows=rows)
        
        # Store in session
        session_data = get_session_data()
        session_data['df'] = df_gen
        session_data['feature_cols'] = ['hour', 'day_of_week', 'is_weekend'] + [c for c in df_gen.columns if '_lag' in c]
        session_data['dish_cols'] = [c for c in df_gen.columns if c not in ['order_placed_at', 'hour', 'day_of_week', 'is_weekend'] and '_lag' not in c]
        
        return jsonify({
            "status": "generated",
            "generated_rows": len(df_gen),
            "dish_columns": session_data['dish_cols'],
            "model_loaded": session_data['model'] is not None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_csv", methods=["GET"])
def get_csv():
    """Return the current session's CSV data"""
    session_data = get_session_data()
    if session_data['df'] is None:
        return "No data available", 404
    
    csv_content = session_data['df'].to_csv(index=False)
    return csv_content, 200, {'Content-Type': 'text/csv'}

@app.route("/train_session", methods=["POST"])
def train_session():
    """
    Train model for the current session using CSV content from client.
    """
    data = request.get_json(silent=True)
    if not data or 'csv_content' not in data:
        return jsonify({"error": "csv_content required"}), 400
    
    csv_content = data['csv_content']
    
    try:
        # Parse CSV content
        df = pd.read_csv(io.StringIO(csv_content))
        
        # Save to temp file for training
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".csv", newline='') as tmp:
            tmp_path = tmp.name
            df.to_csv(tmp_path, index=False)
        
        # Train model
        session_id = get_session_id()
        model_path = f"./model/model_{session_id}.pkl"
        
        model, df_processed, feature_cols, dish_cols = train_demand_model(tmp_path, save_path=model_path)
        
        # Store in session
        session_data = get_session_data()
        session_data['model'] = model
        session_data['df'] = df_processed
        session_data['feature_cols'] = feature_cols
        session_data['dish_cols'] = dish_cols
        
        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        
        return jsonify({
            "status": "trained",
            "training_rows": len(df_processed),
            "dish_columns": dish_cols,
            "feature_cols": feature_cols
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_session/next_hour", methods=["POST"])
def predict_session_next_hour():
    """
    Predict next hour using current session's model and data.
    Accepts csv_content to ensure we're using the latest data.
    """
    session_data = get_session_data()
    
    if session_data['model'] is None:
        return jsonify({"error": "model not loaded. Train a model first."}), 400
    
    try:
        # Use the stored dataframe from session
        df = session_data['df']
        model = session_data['model']
        feature_cols = session_data['feature_cols']
        dish_cols = session_data['dish_cols']
        
        result = predict_next_hour(model, df, feature_cols, dish_cols)
        
        # Convert to native types
        result = {k: (int(v) if hasattr(v, "__int__") else float(v)) for k, v in result.items()}
        
        return jsonify({
            "timestamp": str(df['order_placed_at'].max() + pd.Timedelta(hours=1)),
            "prediction": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_session/next_24_hours", methods=["POST"])
def predict_session_next_24_hours():
    """
    Predict next 24 hours using current session's model and data.
    """
    session_data = get_session_data()
    
    if session_data['model'] is None:
        return jsonify({"error": "model not loaded. Train a model first."}), 400
    
    try:
        df = session_data['df']
        model = session_data['model']
        feature_cols = session_data['feature_cols']
        dish_cols = session_data['dish_cols']
        
        pred_df = predict_next_24_hours(model, df, feature_cols, dish_cols)
        
        # Convert to native types
        records = []
        for r in pred_df.to_dict(orient="records"):
            rec = {}
            for k, v in r.items():
                if k == "timestamp":
                    rec[k] = pd.to_datetime(v).isoformat()
                else:
                    try:
                        rec[k] = int(v)
                    except Exception:
                        try:
                            rec[k] = float(v)
                        except Exception:
                            rec[k] = v
            records.append(rec)
        
        return jsonify({"predictions": records})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/static/<path:filename>")
def static_file(filename):
    root = pathlib.Path(__file__).parent
    static_dir = root / "static"
    if static_dir.exists():
        return send_from_directory(str(static_dir), filename)
    try:
        return app.send_static_file(filename)
    except Exception:
        return jsonify({"error": "static file not found"}), 404

@app.route("/favicon.ico")
def favicon():
    root = pathlib.Path(__file__).parent
    static_dir = root / "static"
    ico_name = "favicon.ico"
    if (static_dir / ico_name).exists():
        return send_from_directory(str(static_dir), ico_name)
    try:
        return app.send_static_file(ico_name)
    except Exception:
        return ("", 204)

@app.route("/", methods=["GET"])
def index():
    try:
        return render_template("index.html")
    except Exception:
        root = pathlib.Path(__file__).parent
        index_path = root / "index.html"
        if index_path.exists():
            return send_from_directory(str(root), "index.html")
        return ("<html><body><h1>Index</h1><p>No index.html found.</p></body></html>", 200)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=int(os.environ.get("PORT", 5000)))