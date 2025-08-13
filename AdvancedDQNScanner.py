#!/usr/bin/env python3
"""
Advanced Web Vulnerability Analysis System with AI (DQN)
Sophisticated educational tool for intelligent defensive security analysis.

Author: Gemini
Version: 3.0
"""

# --- Standard Library Imports ---
import argparse
import json
import logging
import random
import re
import socket
import ssl
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, parse_qs

# --- Third-Party Imports ---
import numpy as np
import requests
import tensorflow as tf
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry
# Desabilitar avisos de SSL para testes em ambientes controlados
from urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# --- Configuration Constants ---
class Config:
    USER_AGENT = 'AdvancedSecurityScanner/3.0-DQN (Educational Purpose)'
    REQUEST_TIMEOUT = 10
    MAX_CRAWL_URLS = 50
    MAX_THREADS = 10
    
    SQLI_PAYLOADS = ["' OR 1=1--", "' OR '1'='1", '" OR 1=1--']
    XSS_PAYLOADS = ["<script>alert('XSS-Test-by-Tool')</script>", "<img src=x onerror=alert('XSS-Test-by-Tool')>"]
    SENSITIVE_PATHS = ['.git/config', '.env', 'Dockerfile', 'docker-compose.yml', '/.aws/credentials', '/WEB-INF/web.xml', '/robots.txt', '/sitemap.xml']
    VULNERABLE_SOFTWARE = {'jquery': lambda v: v < '3.5.0'}

# --- Data Structures ---
@dataclass
class Vulnerability:
    """Class for representing a found vulnerability."""
    category: str
    severity: str
    description: str
    location: str
    recommendation: str
    evidence: Optional[str] = None

# --- Core Web Analyzer Engine ---
class WebVulnerabilityAnalyzer:
    """The core engine for performing web analysis tasks."""
    def __init__(self, target_url: str):
        self.target_url = target_url.rstrip('/')
        self.base_domain = urlparse(target_url).netloc
        self.session = self._create_session()
        self.scanned_urls = set()
        self.lock = threading.Lock()
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({'User-Agent': Config.USER_AGENT})
        session.verify = False # Simplifica testes em ambientes locais/de dev
        return session

    def _get_page_content(self, url: str) -> Optional[requests.Response]:
        try:
            return self.session.get(url, timeout=Config.REQUEST_TIMEOUT)
        except requests.RequestException as e:
            self.logger.warning(f"Could not fetch {url}: {e}")
            return None

    def find_links(self, url: str) -> Set[str]:
        response = self._get_page_content(url)
        if not response: return set()
        soup = BeautifulSoup(response.content, 'html.parser')
        links = set()
        for tag in soup.find_all(['a', 'link', 'script'], href=True):
            href = tag.get('href')
            if href and not href.startswith(('mailto:', 'tel:')):
                full_url = urljoin(url, href)
                if urlparse(full_url).netloc == self.base_domain:
                    links.add(full_url.split('#')[0])
        return links

    def analyze_http_headers(self, url: str) -> List[Vulnerability]:
        vulns = []
        try:
            response = self.session.get(url, timeout=5)
            headers = response.headers
            security_headers = {'Strict-Transport-Security': ('HSTS not configured', 'MEDIUM'), 'X-Content-Type-Options': ('MIME sniffing protection absent', 'LOW'), 'X-Frame-Options': ('Clickjacking protection absent', 'MEDIUM'), 'Content-Security-Policy': ('Content Security Policy not configured', 'HIGH')}
            for header, (message, severity) in security_headers.items():
                if header not in headers:
                    vulns.append(Vulnerability('HTTP Headers', severity, message, url, f'Add {header} header'))
        except requests.RequestException: pass
        return vulns

    def analyze_forms_csrf(self, url: str) -> List[Vulnerability]:
        vulns = []
        response = self._get_page_content(url)
        if not response: return vulns
        soup = BeautifulSoup(response.content, 'html.parser')
        for form in soup.find_all('form'):
            if form.get('method', 'get').upper() == 'POST' and not form.find('input', {'name': re.compile(r'csrf_token|token|nonce', re.I)}):
                vulns.append(Vulnerability('CSRF', 'HIGH', 'Form lacks CSRF token', url, 'Implement anti-CSRF tokens'))
        return vulns

    def analyze_sqli(self, url: str) -> List[Vulnerability]:
        vulns = []
        parsed_url = urlparse(url)
        params = parse_qs(parsed_url.query)
        for param in params:
            original_value = params[param][0]
            for payload in Config.SQLI_PAYLOADS:
                try:
                    test_url = url.replace(f"{param}={original_value}", f"{param}={payload}")
                    response = self.session.get(test_url, timeout=5)
                    if any(e in response.text.lower() for e in ['sql syntax', 'mysql_fetch', 'unclosed quotation']):
                        vulns.append(Vulnerability('SQL Injection', 'CRITICAL', 'Potential SQLi in URL parameter', url, 'Use prepared statements', f'Param: {param}, Payload: {payload}'))
                        break
                except requests.RequestException: pass
        return vulns

    def analyze_xss(self, url: str) -> List[Vulnerability]:
        vulns = []
        response = self._get_page_content(url)
        if not response or not response.text: return vulns
        soup = BeautifulSoup(response.content, 'html.parser')
        for form in soup.find_all('form'):
            inputs = form.find_all(['input', 'textarea'])
            data = {i.get('name'): 'test' for i in inputs if i.get('name')}
            action_url = urljoin(url, form.get('action', ''))
            for payload in Config.XSS_PAYLOADS:
                for input_name in data:
                    test_data = data.copy()
                    test_data[input_name] = payload
                    try:
                        res = self.session.post(action_url, data=test_data, timeout=5) if form.get('method', 'get').lower() == 'post' else self.session.get(action_url, params=test_data, timeout=5)
                        if payload in res.text:
                            vulns.append(Vulnerability('XSS', 'HIGH', 'Reflected XSS in form', action_url, 'Sanitize user input on server-side', f'Field: {input_name}, Payload: {payload}'))
                    except requests.RequestException: pass
        return vulns

    def check_sensitive_files(self) -> List[Vulnerability]:
        vulns = []
        for path in Config.SENSITIVE_PATHS:
            test_url = urljoin(self.target_url, path)
            try:
                response = self.session.get(test_url, timeout=5)
                if response.status_code == 200 and not "not found" in response.text.lower():
                    vulns.append(Vulnerability('Sensitive Data Exposure', 'HIGH', f'Accessible sensitive path found: {path}', test_url, 'Restrict access to this path.', f'Status: {response.status_code}'))
            except requests.RequestException: pass
        return vulns

    def analyze_software_versions(self, url: str) -> List[Vulnerability]:
        vulns = []
        response = self._get_page_content(url)
        if not response: return vulns
        # Check for jQuery versions in script tags
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup.find_all("script", src=True):
            src = script['src']
            match = re.search(r'jquery-([0-9]+\.[0-9]+\.[0-9]+)', src)
            if match:
                version = match.group(1)
                if Config.VULNERABLE_SOFTWARE['jquery'](version):
                    vulns.append(Vulnerability('Outdated Software', 'MEDIUM', f'Vulnerable jQuery version detected: {version}', url, f'Upgrade to jQuery 3.5.0 or newer.', f'Detected in: {src}'))
        return vulns

    def run_all_scans_for_url(self, url: str) -> List[Vulnerability]:
        """Runs all checks concurrently for a single URL."""
        all_vulns = []
        scan_functions = [self.analyze_http_headers, self.analyze_forms_csrf, self.analyze_sqli, self.analyze_xss, self.analyze_software_versions]
        with ThreadPoolExecutor(max_workers=len(scan_functions)) as executor:
            future_to_func = {executor.submit(func, url): func for func in scan_functions}
            for future in as_completed(future_to_func):
                try:
                    result = future.result()
                    if result: all_vulns.extend(result)
                except Exception as exc:
                    self.logger.error(f'A scan function generated an exception: {exc}')
        return all_vulns

# --- Reinforcement Learning Environment ---
class WebAppEnv:
    """The Environment representing the website for the RL agent."""
    def __init__(self, start_url: str):
        self.start_url = start_url
        self.analyzer = WebVulnerabilityAnalyzer(start_url)
        self.known_links = {start_url}
        self.unscanned_links = deque([start_url])
        self.found_vulnerabilities = []
        self.current_url = start_url

        # Ações: 0:Scan Headers, 1:Scan CSRF, 2:Scan SQLi, 3:Scan XSS, 4:Scan Versions, 5:Explore Link
        self.action_space_size = 6
        # Estado: [tem_params, tem_forms, tem_input_pass, num_vulns_total, profundidade_url]
        self.state_space_size = 5

    def reset(self):
        self.known_links = {self.start_url}
        self.unscanned_links = deque([self.start_url])
        self.found_vulnerabilities = []
        self.current_url = self.start_url
        return self._get_state_for_url(self.current_url)

    def _get_state_for_url(self, url: str) -> np.ndarray:
        state = np.zeros(self.state_space_size)
        state[3] = len(self.found_vulnerabilities)
        state[4] = url.count('/') - urlparse(self.start_url).path.count('/')
        
        try:
            response = self.analyzer.session.get(url, timeout=5)
            soup = BeautifulSoup(response.content, 'html.parser')
            parsed_url = urlparse(url)
            state[0] = 1 if parsed_url.query else 0
            state[1] = 1 if soup.find('form') else 0
            state[2] = 1 if soup.find('input', {'type': 'password'}) else 0
        except: pass # Se a página não puder ser obtida, o estado permanece zero
        return state.reshape(1, self.state_space_size)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, List[Vulnerability]]:
        reward = -0.5 # Custo pequeno por ação
        done = False
        new_vulns = []
        
        action_map = {
            0: self.analyzer.analyze_http_headers, 1: self.analyzer.analyze_forms_csrf,
            2: self.analyzer.analyze_sqli, 3: self.analyzer.analyze_xss,
            4: self.analyzer.analyze_software_versions
        }

        if action in action_map:
            new_vulns = action_map[action](self.current_url)
        elif action == 5: # Explore Link
            if self.unscanned_links:
                self.current_url = self.unscanned_links.popleft()
                new_found_links = self.analyzer.find_links(self.current_url)
                for link in new_found_links:
                    if link not in self.known_links:
                        self.known_links.add(link)
                        self.unscanned_links.append(link)
                reward = 1 # Recompensa por explorar
            else:
                reward = -10 # Penalidade grande por tentar explorar sem ter para onde ir
                done = True

        if new_vulns:
            reward_map = {'CRITICAL': 100, 'HIGH': 50, 'MEDIUM': 20, 'LOW': 5}
            for v in new_vulns:
                reward += reward_map.get(v.severity, 1)
                self.found_vulnerabilities.append(v)

        if not self.unscanned_links or len(self.known_links) > Config.MAX_CRAWL_URLS:
            done = True
            
        next_state = self._get_state_for_url(self.current_url)
        return next_state, reward, done, new_vulns

# --- DQN Agent ---
class DQNAgent:
    """The AI Agent that learns the best action policy."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size: return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# --- Reporting ---
def generate_html_report(report_data: dict, filename: str):
    """Generates a professional HTML report."""
    html = f"""
    <html>
    <head>
        <title>Vulnerability Scan Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .severity-CRITICAL {{ background-color: #ff0000; color: white; }}
            .severity-HIGH {{ background-color: #ff4500; }}
            .severity-MEDIUM {{ background-color: #ffc107; }}
            .severity-LOW {{ background-color: #d1ecf1; }}
        </style>
    </head>
    <body>
        <h1>Vulnerability Scan Report</h1>
        <p><strong>Target:</strong> {report_data['target']}</p>
        <p><strong>Scan Date:</strong> {report_data['scan_date']}</p>
        <h2>Summary</h2>
        <p>Total Vulnerabilities Found: {len(report_data['vulnerabilities'])}</p>
        <h2>Details</h2>
        <table>
            <tr>
                <th>Severity</th>
                <th>Category</th>
                <th>Description</th>
                <th>Location</th>
                <th>Recommendation</th>
                <th>Evidence</th>
            </tr>
    """
    for vuln in report_data['vulnerabilities']:
        html += f"""
            <tr>
                <td class="severity-{vuln['severity']}">{vuln['severity']}</td>
                <td>{vuln['category']}</td>
                <td>{vuln['description']}</td>
                <td>{vuln['location']}</td>
                <td>{vuln['recommendation']}</td>
                <td>{vuln.get('evidence', 'N/A')}</td>
            </tr>
        """
    html += """
        </table>
    </body>
    </html>
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[*] HTML report saved to {filename}")

# --- Main Functions for CLI Modes ---
def train_agent(target_url: str, episodes: int, model_path: str):
    """Trains the DQN agent."""
    env = WebAppEnv(target_url)
    agent = DQNAgent(state_size=env.state_space_size, action_size=env.action_space_size)
    batch_size = 32

    print(f"[*] Starting training for {episodes} episodes...")
    for e in tqdm(range(episodes), desc="Training Progress"):
        state = env.reset()
        for time_step in range(150): # Max steps per episode
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done: break
        agent.replay(batch_size)
    
    agent.save(model_path)
    print(f"[+] Training complete. Model saved to {model_path}")

def run_scan(target_url: str, model_path: str, output_prefix: str):
    """Runs a fast scan using a pre-trained agent."""
    print(f"[*] Starting intelligent scan on {target_url}")
    env = WebAppEnv(target_url)
    agent = DQNAgent(state_size=env.state_space_size, action_size=env.action_space_size)
    try:
        agent.load(model_path)
        agent.epsilon = 0.0 # No exploration in scan mode
        print("[*] Pre-trained model loaded successfully.")
    except Exception as e:
        print(f"[-] Could not load model: {e}. Running with random policy.")
        agent.epsilon = 1.0 # Fallback to random actions

    state = env.reset()
    all_found_vulns = set()
    
    # First, run one-off checks like sensitive files
    initial_vulns = env.analyzer.check_sensitive_files()
    for v in initial_vulns: all_found_vulns.add(v.description)
    
    print("[*] Beginning guided crawl and analysis...")
    pbar = tqdm(total=Config.MAX_CRAWL_URLS, desc="Scanning URLs")
    while len(env.analyzer.scanned_urls) < Config.MAX_CRAWL_URLS:
        action = agent.act(state)
        next_state, _, done, new_vulns = env.step(action)
        
        for v in new_vulns:
            if v.description not in all_found_vulns:
                tqdm.write(f"  [+] Found: {v.category} ({v.severity}) at {v.location}")
                all_found_vulns.add(v.description)

        state = next_state
        pbar.update(len(env.analyzer.scanned_urls) - pbar.n)
        if done: break
    pbar.close()

    # Final report generation
    final_vulnerabilities = [asdict(v) for v in env.found_vulnerabilities]
    report_data = {
        'target': target_url,
        'scan_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'vulnerabilities': final_vulnerabilities
    }
    
    json_filename = f"{output_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    html_filename = f"{output_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)
    print(f"[*] JSON report saved to {json_filename}")
    generate_html_report(report_data, html_filename)

def main():
    parser = argparse.ArgumentParser(
        description="Advanced Web Vulnerability Scanner using Deep Q-Learning.",
        epilog="Use 'train' to build a model, then 'scan' to use it. Always 