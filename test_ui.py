import sys
import time
from playwright.sync_api import sync_playwright

def run() -> None:
    print("Iniciando prueba local de UI con Playwright...")
    with sync_playwright() as p:
        # Launch headless Chromium
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()
        
        try:
            print("Navegando a http://localhost:4321...")
            page.goto("http://localhost:4321", timeout=15000)
            
            # Wait for elements to render
            page.wait_for_selector("#drop-zone", timeout=10000)
            print("✔ UI cargada correctamente. Elemento #drop-zone encontrado.")
            
            # Give a small margin for fonts/styles loading
            time.sleep(2)
            
            # Take screenshot and save it to the conversation artifacts directory
            screenshot_path = "/home/dracero/.gemini/antigravity-ide/brain/aa7be603-3969-482f-92d6-c37fd02fe0af/ui_screenshot.png"
            page.screenshot(path=screenshot_path, full_page=False)
            print(f"✔ Captura de pantalla guardada en: {screenshot_path}")
            
        except Exception as e:
            print(f"❌ Error durante la prueba de la interfaz: {e}")
            sys.exit(1)
        finally:
            browser.close()

if __name__ == "__main__":
    run()
