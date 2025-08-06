#!/usr/bin/env python3
"""
Convert markdown documentation to PDF using built-in libraries
"""

import re
import html
import subprocess
import os

def markdown_to_html(markdown_text):
    """Convert markdown to HTML using basic regex replacements"""
    
    # Escape HTML characters first
    html_text = html.escape(markdown_text)
    
    # Convert headers
    html_text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html_text, flags=re.MULTILINE)
    html_text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html_text, flags=re.MULTILINE)
    html_text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html_text, flags=re.MULTILINE)
    html_text = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html_text, flags=re.MULTILINE)
    
    # Convert code blocks
    html_text = re.sub(r'```python\n(.*?)\n```', r'<pre><code class="python">\1</code></pre>', html_text, flags=re.DOTALL)
    html_text = re.sub(r'```\n(.*?)\n```', r'<pre><code>\1</code></pre>', html_text, flags=re.DOTALL)
    html_text = re.sub(r'`([^`]+)`', r'<code>\1</code>', html_text)
    
    # Convert bold and italic
    html_text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_text)
    html_text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html_text)
    
    # Convert lists
    html_text = re.sub(r'^- (.+)$', r'<li>\1</li>', html_text, flags=re.MULTILINE)
    html_text = re.sub(r'^(\d+)\. (.+)$', r'<li>\2</li>', html_text, flags=re.MULTILINE)
    
    # Wrap consecutive list items in ul tags
    html_text = re.sub(r'(<li>.*?</li>)(?:\n<li>.*?</li>)*', r'<ul>\g<0></ul>', html_text, flags=re.DOTALL)
    
    # Convert line breaks
    html_text = html_text.replace('\n\n', '</p><p>')
    html_text = '<p>' + html_text + '</p>'
    
    # Clean up empty paragraphs
    html_text = re.sub(r'<p>\s*</p>', '', html_text)
    
    return html_text

def create_html_document(content):
    """Create a complete HTML document with styling"""
    
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Algorithmic Trading System Documentation</title>
    <style>
        body {{
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        h3 {{
            color: #5d6d7e;
        }}
        code {{
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #3498db;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
        }}
        ul {{
            margin: 10px 0;
        }}
        li {{
            margin: 5px 0;
        }}
        .abstract {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
            font-style: italic;
        }}
        @media print {{
            body {{
                max-width: none;
                margin: 0;
                padding: 15px;
            }}
        }}
    </style>
</head>
<body>
{content}
</body>
</html>
"""
    return html_template

def main():
    # Read the markdown file
    with open('/home/sid12321/Desktop/Trading-Final/TECHNICAL_DOCUMENTATION.md', 'r') as f:
        markdown_content = f.read()
    
    # Convert to HTML
    html_content = markdown_to_html(markdown_content)
    
    # Create complete HTML document
    full_html = create_html_document(html_content)
    
    # Save HTML file
    html_file = '/home/sid12321/Desktop/Trading-Final/TECHNICAL_DOCUMENTATION.html'
    with open(html_file, 'w') as f:
        f.write(full_html)
    
    print(f"HTML file created: {html_file}")
    
    # Try to convert to PDF using wkhtmltopdf if available
    try:
        pdf_file = '/home/sid12321/Desktop/Trading-Final/TECHNICAL_DOCUMENTATION.pdf'
        subprocess.run(['wkhtmltopdf', '--page-size', 'A4', '--margin-top', '0.75in', 
                       '--margin-right', '0.75in', '--margin-bottom', '0.75in', 
                       '--margin-left', '0.75in', html_file, pdf_file], 
                      check=True, capture_output=True)
        print(f"PDF file created: {pdf_file}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("wkhtmltopdf not available, trying alternative method...")
        
    # Try using chromium-browser if available
    try:
        pdf_file = '/home/sid12321/Desktop/Trading-Final/TECHNICAL_DOCUMENTATION.pdf'
        subprocess.run(['chromium-browser', '--headless', '--disable-gpu', '--print-to-pdf=' + pdf_file,
                       '--no-margins', html_file], 
                      check=True, capture_output=True)
        print(f"PDF file created using Chromium: {pdf_file}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Chromium not available, trying Firefox...")
        
    # Try using firefox if available
    try:
        pdf_file = '/home/sid12321/Desktop/Trading-Final/TECHNICAL_DOCUMENTATION.pdf'
        subprocess.run(['firefox', '--headless', '--print-to-pdf=' + pdf_file, html_file], 
                      check=True, capture_output=True)
        print(f"PDF file created using Firefox: {pdf_file}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Firefox not available either.")
        
    print("Could not convert to PDF automatically. HTML file is available for manual conversion.")
    print("You can:")
    print("1. Open the HTML file in a browser and use 'Print to PDF'")
    print("2. Install wkhtmltopdf: sudo apt install wkhtmltopdf")
    print("3. Use an online markdown to PDF converter")
    
    return False

if __name__ == "__main__":
    main()