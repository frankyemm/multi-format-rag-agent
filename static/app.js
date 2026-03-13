/**
 * RAG AI Data Analyst — Frontend Logic
 *
 * Handles:
 * - File upload with drag & drop and progress feedback
 * - Chat with question/answer via the /ask API
 * - Document management (list, delete)
 * - Markdown rendering for responses
 */

// ============================================================
// DOM References
// ============================================================

const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const uploadProgress = document.getElementById('upload-progress');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const documentsList = document.getElementById('documents-list');
const noDocs = document.getElementById('no-docs');
const chatMessages = document.getElementById('chat-messages');
const welcomeScreen = document.getElementById('welcome-screen');
const typingIndicator = document.getElementById('typing-indicator');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const statusText = document.getElementById('status-text');
const mobileToggle = document.getElementById('mobile-toggle');
const sidebar = document.getElementById('sidebar');
const mobileOverlay = document.getElementById('mobile-overlay');

// ============================================================
// State
// ============================================================

let isProcessing = false;
let documents = [];

// ============================================================
// Initialization
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    loadDocuments();
    setupUpload();
    setupChat();
    setupMobile();
    setupHints();
});

// ============================================================
// Upload Logic
// ============================================================

function setupUpload() {
    // Click to upload
    uploadZone.addEventListener('click', () => fileInput.click());

    // File selection
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFiles(Array.from(e.target.files));
        }
    });

    // Drag & drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFiles(Array.from(e.dataTransfer.files));
        }
    });
}

async function handleFiles(files) {
    for (const file of files) {
        await uploadFile(file);
    }
}

async function uploadFile(file) {
    // Show progress
    uploadProgress.classList.add('active');
    progressFill.style.width = '10%';
    progressText.textContent = `Subiendo "${file.name}"...`;
    statusText.textContent = 'Procesando...';

    const formData = new FormData();
    formData.append('file', file);

    try {
        // Simulate progress
        progressFill.style.width = '30%';
        progressText.textContent = `Extrayendo texto de "${file.name}"...`;

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
        });

        progressFill.style.width = '70%';
        progressText.textContent = 'Creando embeddings...';

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Error al subir el archivo');
        }

        progressFill.style.width = '100%';
        progressText.textContent = `✓ ${data.message}`;

        // Refresh document list
        await loadDocuments();

        // Show success briefly
        setTimeout(() => {
            uploadProgress.classList.remove('active');
            progressFill.style.width = '0%';
        }, 2000);

    } catch (error) {
        progressFill.style.width = '100%';
        progressFill.style.background = 'var(--danger)';
        progressText.textContent = `✗ Error: ${error.message}`;

        setTimeout(() => {
            uploadProgress.classList.remove('active');
            progressFill.style.width = '0%';
            progressFill.style.background = '';
        }, 3000);
    }

    statusText.textContent = 'Listo';
    fileInput.value = '';
}

// ============================================================
// Documents Management
// ============================================================

async function loadDocuments() {
    try {
        const response = await fetch('/documents');
        const data = await response.json();
        documents = data.documents || [];
        renderDocuments();
    } catch (error) {
        console.error('Error loading documents:', error);
    }
}

function renderDocuments() {
    // Clear existing (keep noDocs element)
    const existing = documentsList.querySelectorAll('.doc-item');
    existing.forEach(el => el.remove());

    if (documents.length === 0) {
        noDocs.style.display = 'block';
        return;
    }

    noDocs.style.display = 'none';

    documents.forEach(doc => {
        const ext = doc.extension.replace('.', '');
        const iconMap = {
            pdf: '📕',
            docx: '📘',
            xlsx: '📗',
            xls: '📗',
            txt: '📄',
        };

        const size = formatFileSize(doc.size_bytes);

        const item = document.createElement('div');
        item.className = 'doc-item';
        item.innerHTML = `
            <div class="doc-icon ${ext}">${iconMap[ext] || '📄'}</div>
            <div class="doc-info">
                <div class="doc-name" title="${doc.filename}">${doc.filename}</div>
                <div class="doc-meta">${size} · ${doc.num_chunks} fragmentos</div>
            </div>
            <button class="doc-delete" title="Eliminar" data-id="${doc.doc_id}">🗑️</button>
        `;

        item.querySelector('.doc-delete').addEventListener('click', (e) => {
            e.stopPropagation();
            deleteDocument(doc.doc_id, doc.filename);
        });

        documentsList.appendChild(item);
    });
}

async function deleteDocument(docId, filename) {
    if (!confirm(`¿Eliminar "${filename}" y todos sus datos vectorizados?`)) return;

    try {
        const response = await fetch(`/documents/${docId}`, { method: 'DELETE' });
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.detail || 'Error al eliminar');
        }
        await loadDocuments();
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

function formatFileSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(1)} MB`;
}

// ============================================================
// Chat Logic
// ============================================================

function setupChat() {
    // Send button
    sendBtn.addEventListener('click', sendMessage);

    // Enter to send, Shift+Enter for new line
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
    });
}

async function sendMessage() {
    const question = chatInput.value.trim();
    if (!question || isProcessing) return;

    // Hide welcome screen
    if (welcomeScreen) {
        welcomeScreen.style.display = 'none';
    }

    // Add user message
    addMessage(question, 'user');

    // Clear input
    chatInput.value = '';
    chatInput.style.height = 'auto';

    // Show typing indicator
    isProcessing = true;
    sendBtn.disabled = true;
    typingIndicator.classList.add('active');
    statusText.textContent = 'Analizando...';
    scrollToBottom();

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Error al obtener respuesta');
        }

        // Hide typing indicator and show response
        typingIndicator.classList.remove('active');
        addMessage(data.answer, 'assistant', data.sources);

    } catch (error) {
        typingIndicator.classList.remove('active');
        addMessage(`⚠️ Error: ${error.message}`, 'assistant');
    }

    isProcessing = false;
    sendBtn.disabled = false;
    statusText.textContent = 'Listo';
    chatInput.focus();
}

function addMessage(content, role, sources = []) {
    const msg = document.createElement('div');
    msg.className = `message ${role}`;

    const avatar = role === 'assistant' ? '🧠' : '👤';
    let html = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            ${role === 'assistant' ? renderMarkdown(content) : `<p>${escapeHtml(content)}</p>`}
    `;

    if (sources && sources.length > 0) {
        html += `
            <div class="message-sources">
                ${sources.map(s => `<span class="source-tag">📎 ${escapeHtml(s)}</span>`).join('')}
            </div>
        `;
    }

    html += `</div>`;
    msg.innerHTML = html;

    // Insert before typing indicator
    chatMessages.insertBefore(msg, typingIndicator);
    scrollToBottom();
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });
}

// ============================================================
// Hint Chips
// ============================================================

function setupHints() {
    document.querySelectorAll('.hint-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            chatInput.value = chip.dataset.hint;
            chatInput.focus();
        });
    });
}

// ============================================================
// Mobile Navigation
// ============================================================

function setupMobile() {
    mobileToggle.addEventListener('click', () => {
        sidebar.classList.toggle('open');
        mobileOverlay.classList.toggle('active');
    });

    mobileOverlay.addEventListener('click', () => {
        sidebar.classList.remove('open');
        mobileOverlay.classList.remove('active');
    });
}

// ============================================================
// Markdown Renderer (lightweight)
// ============================================================

function renderMarkdown(text) {
    if (!text) return '';

    let html = text;

    // Handle Pensamiento (Internal Monologue) first
    // We do this before escaping so we can use the tags
    html = html.replace(/&lt;pensamiento&gt;([\s\S]*?)&lt;\/pensamiento&gt;/g, (match, content) => {
        return `<details class="thought-container" open>
            <summary class="thought-header">Monólogo Interno de Análisis</summary>
            <div class="thought-content">${content.trim()}</div>
        </details>`;
    });

    // If it hasn't been escaped yet, let's be careful.
    // Actually, I'll move the pensée handling AFTER initial escaping but before other markdown tags.
    
    // Let's restart the function logic for clarity.
    return processMarkdown(text);
}

function processMarkdown(text) {
    let html = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Handle Pensamiento (Internal Monologue)
    html = html.replace(/&lt;pensamiento&gt;([\s\S]*?)&lt;\/pensamiento&gt;/g, (match, content) => {
        // Unescape internal content slightly for basic formatting if needed, but safe
        return `<details class="thought-container">
            <summary class="thought-header">Monólogo Interno de Análisis</summary>
            <div class="thought-content">${content.trim().replace(/\n/g, '<br>')}</div>
        </details>`;
    });

    // Code blocks (``` ... ```)
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
        return `<pre><code>${code.trim()}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Headers
    html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Italic
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Tables: detect markdown tables and convert them
    html = html.replace(/^(\|.+\|)\n(\|[-| :]+\|)\n((?:\|.+\|\n?)*)/gm, (match, headerRow, separator, bodyRows) => {
        const headers = headerRow.split('|').filter(c => c.trim()).map(c => `<th>${c.trim()}</th>`).join('');
        const rows = bodyRows.trim().split('\n').map(row => {
            const cells = row.split('|').filter(c => c.trim()).map(c => `<td>${c.trim()}</td>`).join('');
            return `<tr>${cells}</tr>`;
        }).join('');
        return `<table><thead><tr>${headers}</tr></thead><tbody>${rows}</tbody></table>`;
    });

    // Unordered lists
    html = html.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/g, (match) => `<ul>${match}</ul>`);

    // Ordered lists
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Line breaks → paragraphs
    html = html.replace(/\n\n+/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');

    // Wrap in paragraph if not already wrapped
    if (!html.startsWith('<')) {
        html = `<p>${html}</p>`;
    } else if (!html.startsWith('<p>')) {
        html = `<p>${html}</p>`;
    }

    // Clean up empty paragraphs
    html = html.replace(/<p>\s*<\/p>/g, '');
    html = html.replace(/<p>(<h[1-6]>)/g, '$1');
    html = html.replace(/(<\/h[1-6]>)<\/p>/g, '$1');
    html = html.replace(/<p>(<pre>)/g, '$1');
    html = html.replace(/(<\/pre>)<\/p>/g, '$1');
    html = html.replace(/<p>(<table>)/g, '$1');
    html = html.replace(/(<\/table>)<\/p>/g, '$1');
    html = html.replace(/<p>(<ul>)/g, '$1');
    html = html.replace(/(<\/ul>)<\/p>/g, '$1');

    return html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
