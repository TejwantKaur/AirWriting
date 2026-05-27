import streamlit.components.v1 as components
from utils.load_css import load_css

def hero():
    css = load_css([
        "assets/home_styles/base.css",
        "assets/home_styles/hero.css",
    ])

    html = f"""
    <style>
    {css}
    </style>

    <div class="hero">
        <div class="hero-left">
            <h1>AIR WRITING</h1>
            <p>Start interacting with us in air 😊</p>
            <p>
                Experience futuristic gesture-based writing
                using computer vision and artificial intelligence.
            </p>
            <div class="hero-btn">
                Get Started
            </div>
        </div>

        <div class="hero-right">
            <div class="glass-card">
                <h3>✨ Smart Recognition</h3>
                <p>
                    Real-time gesture tracking and handwriting prediction.
                </p>
            </div>
        </div>
    </div>
    """
    components.html(html, height=500)