from flask import Flask, jsonify, Response
from flask_smorest import Api
from flask_cors import CORS
from mancala_ai.api.routes import bp

SWAGGER_CSS = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/swagger-ui.css"
SWAGGER_BUNDLE = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/swagger-ui-bundle.js"
SWAGGER_STANDALONE = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/swagger-ui-standalone-preset.js"

def create_app():
    app = Flask(__name__)

    # smorest OpenAPI basics (still useful for schema generation)
    app.config["API_TITLE"] = "Mancala AI (Flask)"
    app.config["API_VERSION"] = "v1"
    app.config["OPENAPI_VERSION"] = "3.0.3"

    api = Api(app)               # build spec
    api.register_blueprint(bp)   # your /api/* endpoints
    
    # Allow your UI origins (dev + prod) to call /api/*
    CORS(app, resources={r"/api/*": {
        "origins": [
            "http://localhost:5173",   # dev UI
            "http://127.0.0.1:5173",   # dev UI
            "http://localhost:8080",   # dev UI
            "http://127.0.0.1:8080"
            "https://your-ui-domain.com"  # prod UI
        ]
    }})
    # --- Manual docs: /openapi.json + /apidocs --------------------------------
    @app.get("/openapi.json")
    def openapi_json():
        # api.spec is an APISpec; jsonify the dict form
        return jsonify(api.spec.to_dict())

    @app.get("/apidocs")
    def apidocs():
        html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>Mancala API Docs</title>
    <link rel="stylesheet" href="{SWAGGER_CSS}">
    <style>body {{ margin:0; background:#fafafa; }}</style>
  </head>
  <body>
    <div id="swagger-ui"></div>
    <script src="{SWAGGER_BUNDLE}"></script>
    <script src="{SWAGGER_STANDALONE}"></script>
    <script>
      window.onload = () => {{
        SwaggerUIBundle({{
          url: "/openapi.json",
          dom_id: "#swagger-ui",
          presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset],
          layout: "StandaloneLayout"
        }});
      }};
    </script>
  </body>
</html>"""
        return Response(html, mimetype="text/html")
    # --------------------------------------------------------------------------

    @app.get("/debug/routes")
    def routes():
        return {"routes": [str(r) for r in app.url_map.iter_rules()]}

    return app

if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=8000, debug=True)
