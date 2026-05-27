import streamlit.components.v1 as components
from utils.load_css import load_css

def footer():
    css = load_css([
        "assets/home_styles/base.css",
        "assets/home_styles/footer.css",
    ])

    html = f"""
    <style>
    {css}
    </style>

    <div class="footer">
        Created with ❤️ by Tejwant Kaur
    </div>
    """
    components.html(html, height=80)