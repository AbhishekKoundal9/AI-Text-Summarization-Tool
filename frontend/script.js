document.addEventListener('DOMContentLoaded', () => {
    // Initialize 3D Background - Floating Data Grid / Matrix
    const initDataGridBackground = () => {
        const container = document.getElementById('vanta-bg');
        if (!container || !window.THREE) return;

        container.innerHTML = '';

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0f172a); 
        // Fog creates depth, making points fade out in the distance
        scene.fog = new THREE.FogExp2(0x0f172a, 0.06);

        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        // 1. Create Floating Data Particles (Matrix Rain effect)
        const particleCount = 2500;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const velocities = [];

        for (let i = 0; i < particleCount; i++) {
            positions[i * 3] = (Math.random() - 0.5) * 50;     // x
            positions[i * 3 + 1] = (Math.random() - 0.5) * 50; // y
            positions[i * 3 + 2] = (Math.random() - 0.5) * 50; // z
            velocities.push({
                y: -(0.02 + Math.random() * 0.05) // Speed of falling data
            });
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        // Create square texture for particles to look like data nodes
        const canvas = document.createElement('canvas');
        canvas.width = 16;
        canvas.height = 16;
        const context = canvas.getContext('2d');
        context.fillStyle = '#ffffff';
        context.fillRect(0, 0, 16, 16);
        const texture = new THREE.CanvasTexture(canvas);

        const material = new THREE.PointsMaterial({
            color: 0x6366f1, // Indigo theme color
            size: 0.15,
            map: texture,
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending
        });

        const particles = new THREE.Points(geometry, material);
        scene.add(particles);

        // 2. Create Glowing Floor and Ceiling Grids
        // Floor grid
        const gridHelperBottom = new THREE.GridHelper(80, 80, 0x6366f1, 0x334155);
        gridHelperBottom.position.y = -10;
        gridHelperBottom.material.transparent = true;
        gridHelperBottom.material.opacity = 0.3;
        scene.add(gridHelperBottom);
        
        // Ceiling grid (Purple accent)
        const gridHelperTop = new THREE.GridHelper(80, 80, 0xa855f7, 0x334155);
        gridHelperTop.position.y = 10;
        gridHelperTop.material.transparent = true;
        gridHelperTop.material.opacity = 0.2;
        scene.add(gridHelperTop);

        camera.position.z = 12;
        camera.position.y = 0;

        let time = 0;
        const animate = () => {
            requestAnimationFrame(animate);
            time += 0.005;

            // Animate falling data points
            const positionAttribute = geometry.attributes.position;
            for (let i = 0; i < particleCount; i++) {
                let y = positionAttribute.getY(i);
                y += velocities[i].y;
                
                // Reset to top if it falls below the floor grid
                if (y < -15) {
                    y = 15;
                    // Randomize X and Z slightly when respawning
                    positionAttribute.setX(i, (Math.random() - 0.5) * 50);
                    positionAttribute.setZ(i, (Math.random() - 0.5) * 50);
                }
                positionAttribute.setY(i, y);
            }
            positionAttribute.needsUpdate = true;

            // Move the grids to create infinite scrolling effect
            gridHelperBottom.position.z = (time * 10) % 1;
            gridHelperTop.position.z = (time * 10) % 1;

            // Slow panning camera
            camera.position.x = Math.sin(time) * 4;
            camera.lookAt(scene.position);

            renderer.render(scene, camera);
        };
        
        animate();

        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    };

    initDataGridBackground();

    const summarizeBtn = document.getElementById('summarize-btn');
    const sourceText = document.getElementById('source-text');
    const summaryContent = document.getElementById('summary-content');
    const btnText = document.querySelector('.btn-text');
    const loader = document.querySelector('.loader');
    
    // New Elements
    const fileInput = document.getElementById('file-input');
    const uploadFileBtn = document.getElementById('upload-file-btn');
    const lengthSelect = document.getElementById('length-select');
    const styleSelect = document.getElementById('style-select');
    const lengthContainer = document.getElementById('length-container');
    const bulletContainer = document.getElementById('bullet-container');
    const bulletCountInput = document.getElementById('bullet-count');
    const copyBtn = document.getElementById('copy-btn');
    const exportWordBtn = document.getElementById('export-word-btn');
    const exportPdfBtn = document.getElementById('export-pdf-btn');
    const keywordsContainer = document.getElementById('keywords-container');
    const historyList = document.getElementById('history-list');
    const sentimentBadge = document.querySelector('#sentiment-container span');
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    const toggleHistoryBtn = document.getElementById('toggle-history-btn');
    const closeHistoryBtn = document.getElementById('close-history-btn');
    const historySidebar = document.getElementById('history-sidebar');
    const sidebarOverlay = document.getElementById('sidebar-overlay');

    // Automatically use localhost for development, but prepare a placeholder for the live deployed backend
    const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    const BACKEND_URL = isLocalhost ? 'http://localhost:8000' : 'https://YOUR_LIVE_BACKEND_URL_HERE';

    // State for exports
    let currentSummary = "";
    let currentKeywords = [];
    let currentSentiment = "Neutral";

    // Load History on start
    loadHistory();

    // Drawer Logic
    function toggleDrawer() {
        historySidebar.classList.toggle('open');
        sidebarOverlay.classList.toggle('show');
    }

    toggleHistoryBtn.addEventListener('click', toggleDrawer);
    closeHistoryBtn.addEventListener('click', toggleDrawer);
    sidebarOverlay.addEventListener('click', toggleDrawer);

    // Style Toggle Logic
    styleSelect.addEventListener('change', (e) => {
        if (e.target.value === 'extractive') {
            lengthContainer.style.display = 'none';
            bulletContainer.style.display = 'flex';
        } else {
            lengthContainer.style.display = 'flex';
            bulletContainer.style.display = 'none';
        }
    });

    function updateSentimentBadge(sentiment) {
        sentimentBadge.className = 'badge'; // Reset classes
        if (sentiment === 'Positive') {
            sentimentBadge.classList.add('badge-positive');
        } else if (sentiment === 'Negative') {
            sentimentBadge.classList.add('badge-negative');
        } else {
            sentimentBadge.classList.add('badge-neutral');
        }
        sentimentBadge.innerText = `Sentiment: ${sentiment}`;
    }

    // 1. File Upload Logic
    uploadFileBtn.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        sourceText.value = "Extracting text from file, please wait...";
        summarizeBtn.disabled = true;

        try {
            const response = await fetch(`${BACKEND_URL}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error("Upload failed");
            const data = await response.json();
            sourceText.value = data.text;
        } catch (error) {
            console.error('File extraction error:', error);
            alert("Failed to extract text from file.");
            sourceText.value = "";
        } finally {
            summarizeBtn.disabled = false;
            fileInput.value = ""; // Reset input
        }
    });

    // 1.5 URL Scraping Logic
    const urlInput = document.getElementById('url-input');
    const fetchUrlBtn = document.getElementById('fetch-url-btn');

    fetchUrlBtn.addEventListener('click', async () => {
        const url = urlInput.value.trim();
        if (!url) {
            alert('Please enter a valid URL.');
            return;
        }

        sourceText.value = "Scraping webpage, please wait...";
        summarizeBtn.disabled = true;
        fetchUrlBtn.disabled = true;

        try {
            const response = await fetch(`${BACKEND_URL}/scrape`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: url })
            });

            if (!response.ok) throw new Error("Scrape failed");
            const data = await response.json();
            sourceText.value = data.text;
        } catch (error) {
            console.error('Scraping error:', error);
            alert("Failed to extract text from the provided URL. Ensure it's a valid link.");
            sourceText.value = "";
        } finally {
            summarizeBtn.disabled = false;
            fetchUrlBtn.disabled = false;
            urlInput.value = ""; // Reset input
        }
    });

    // 2. Export and Copy Button Logic
    copyBtn.addEventListener('click', () => {
        const textToCopy = summaryContent.innerText;
        if (textToCopy && textToCopy !== "Your summary will appear here..." && textToCopy !== "Generating summary, please wait...") {
            navigator.clipboard.writeText(textToCopy).then(() => {
                const originalText = copyBtn.innerText;
                copyBtn.innerText = "✅ Copied!";
                setTimeout(() => copyBtn.innerText = originalText, 2000);
            });
        }
    });

    exportWordBtn.addEventListener('click', async () => {
        if (!currentSummary) {
            alert('Please generate a summary first.');
            return;
        }

        const originalText = exportWordBtn.innerText;
        exportWordBtn.innerText = "⏳ Exporting...";
        
        try {
            const response = await fetch(`${BACKEND_URL}/export`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    summary: currentSummary,
                    keywords: currentKeywords,
                    sentiment: currentSentiment
                })
            });

            if (!response.ok) throw new Error("Export failed");
            
            // Download the file
            const blob = await response.blob();
            const downloadUrl = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = downloadUrl;
            a.download = 'Summary.docx';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(downloadUrl);
        } catch (error) {
            console.error('Export Error:', error);
            alert("Failed to export Word document.");
        } finally {
            exportWordBtn.innerText = originalText;
        }
    });

    exportPdfBtn.addEventListener('click', () => {
        if (!currentSummary) {
            alert('Please generate a summary first.');
            return;
        }
        const originalText = exportPdfBtn.innerText;
        exportPdfBtn.innerText = "⏳ Exporting...";

        const element = document.getElementById('output-panel-capture');
        const opt = {
            margin:       0.5,
            filename:     'Summary.pdf',
            image:        { type: 'jpeg', quality: 0.98 },
            html2canvas:  { scale: 2, backgroundColor: '#0f172a' },
            jsPDF:        { unit: 'in', format: 'letter', orientation: 'portrait' }
        };

        html2pdf().set(opt).from(element).save().then(() => {
            exportPdfBtn.innerText = originalText;
        });
    });

    // 3. Summarize Logic
    summarizeBtn.addEventListener('click', async () => {
        const text = sourceText.value.trim();

        if (!text) {
            alert('Please enter some text to summarize.');
            return;
        }

        // Determine length parameters
        let maxLength = 130;
        let minLength = 30;
        if (lengthSelect.value === 'short') {
            maxLength = 60;
            minLength = 15;
        } else if (lengthSelect.value === 'detailed') {
            maxLength = 300;
            minLength = 100;
        }

        const styleParam = styleSelect.value;
        const bulletCount = parseInt(bulletCountInput.value) || 3;

        // UI Loading State
        summarizeBtn.disabled = true;
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        summaryContent.innerHTML = '<p class="placeholder-text">Generating summary, please wait...</p>';
        keywordsContainer.innerHTML = '<span class="placeholder-text" style="font-size: 0.8rem;">Extracting...</span>';
        sentimentBadge.className = 'badge badge-neutral';
        sentimentBadge.innerText = 'Sentiment: Analyzing...';

        try {
            const response = await fetch(`${BACKEND_URL}/summarize`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    max_length: maxLength,
                    min_length: minLength,
                    style: styleParam,
                    bullet_count: bulletCount
                }),
            });

            if (!response.ok) throw new Error(`Server error: ${response.status}`);

            const data = await response.json();
            
            // Display summary (preserve newlines for bullet points)
            summaryContent.innerHTML = `<div style="white-space: pre-wrap;">${data.summary}</div>`;
            
            // Display keywords
            if (data.keywords && data.keywords.length > 0) {
                keywordsContainer.innerHTML = data.keywords.map(kw => `<span class="badge">${kw}</span>`).join('');
            } else {
                keywordsContainer.innerHTML = '<span class="placeholder-text" style="font-size: 0.8rem;">No keywords found</span>';
            }

            // Display Sentiment
            updateSentimentBadge(data.sentiment);
            
            // Update state
            currentSummary = data.summary;
            currentKeywords = data.keywords || [];
            currentSentiment = data.sentiment || "Neutral";

            // Save to history
            saveToHistory(text.substring(0, 50) + "...", data.summary, data.sentiment, data.keywords);

        } catch (error) {
            console.error('Error during summarization:', error);
            summaryContent.innerHTML = `<p style="color: #ef4444;">Failed to generate summary. Make sure the backend server is running.</p>`;
            keywordsContainer.innerHTML = '';
            sentimentBadge.innerText = 'Sentiment: Error';
        } finally {
            // Reset UI State
            summarizeBtn.disabled = false;
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
        }
    });

    // 4. History Logic
    clearHistoryBtn.addEventListener('click', () => {
        if (confirm("Are you sure you want to clear all history?")) {
            localStorage.removeItem('summaryHistory');
            loadHistory();
        }
    });

    function saveToHistory(title, summary, sentiment, keywords) {
        let history = JSON.parse(localStorage.getItem('summaryHistory') || '[]');
        history.unshift({ title, summary, sentiment: sentiment || 'Neutral', keywords: keywords || [], date: new Date().toLocaleString() });
        // Keep only last 10
        if (history.length > 10) history = history.slice(0, 10);
        localStorage.setItem('summaryHistory', JSON.stringify(history));
        renderHistory(history);
    }

    function loadHistory() {
        const history = JSON.parse(localStorage.getItem('summaryHistory') || '[]');
        renderHistory(history);
    }

    function renderHistory(history) {
        if (history.length === 0) {
            historyList.innerHTML = '<p class="placeholder-text">No saved summaries yet.</p>';
            return;
        }

        historyList.innerHTML = history.map((item, index) => {
            let color = '#94a3b8';
            if (item.sentiment === 'Positive') color = '#22c55e';
            if (item.sentiment === 'Negative') color = '#ef4444';
            
            return `
            <div class="history-item" data-index="${index}" style="position: relative;">
                <button class="delete-history-btn" data-index="${index}" style="position: absolute; top: 0.5rem; right: 0.5rem; background: none; border: none; color: #ef4444; cursor: pointer; font-size: 0.8rem;" title="Delete this item">✖</button>
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.3rem; padding-right: 1.5rem;">
                    <span>${item.date}</span>
                    <span style="color: ${color}; font-weight: bold;">${item.sentiment || 'Neutral'}</span>
                </div>
                <strong>${item.title}</strong>
            </div>
            `;
        }).join('');

        // Add click listeners to load history
        document.querySelectorAll('.history-item').forEach(el => {
            el.addEventListener('click', (e) => {
                // Ignore clicks on the delete button
                if (e.target.classList.contains('delete-history-btn')) return;

                const idx = e.currentTarget.getAttribute('data-index');
                const item = history[idx];
                summaryContent.innerHTML = `<div style="white-space: pre-wrap;">${item.summary}</div>`;
                keywordsContainer.innerHTML = '<span class="badge">From History</span>';
                updateSentimentBadge(item.sentiment || 'Neutral');
                sourceText.value = item.title + " (Full text not saved in history to save space)";
                
                // Update state
                currentSummary = item.summary;
                currentKeywords = item.keywords || [];
                currentSentiment = item.sentiment || 'Neutral';
                
                // Close drawer on mobile/desktop
                if (historySidebar.classList.contains('open')) {
                    toggleDrawer();
                }
            });
        });

        // Add click listeners to delete buttons
        document.querySelectorAll('.delete-history-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation(); // Prevent triggering the history item click
                const idx = e.target.getAttribute('data-index');
                history.splice(idx, 1);
                localStorage.setItem('summaryHistory', JSON.stringify(history));
                renderHistory(history);
            });
        });
    }
});
