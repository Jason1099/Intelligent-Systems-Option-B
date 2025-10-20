// Tabs
document.addEventListener('DOMContentLoaded', () => {
    const links = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.content');

    links.forEach(link => {
        link.addEventListener('click', e => {
            e.preventDefault();
            const id = link.dataset.page;
            links.forEach(a => a.classList.remove('active'));
            link.classList.add('active');
            sections.forEach(s => s.classList.toggle('active', s.id === id));
            history.replaceState(null, '', `#${id}`);
        });
    });

    const initial = location.hash.replace('#', '') || 'main';
    document.querySelector(`.nav-link[data-page="${initial}"]`)?.click();
});

// Helpers 
function wireDropzone({ zone, fileInput, previewImg, placeholder }) {
    const openPicker = () => fileInput.click();

    zone.addEventListener('click', openPicker);
    ['dragenter','dragover'].forEach(evt => zone.addEventListener(evt, e => {
        e.preventDefault(); e.stopPropagation(); zone.classList.add('dragging');
    }));
    ['dragleave','drop'].forEach(evt => zone.addEventListener(evt, e => {
        e.preventDefault(); e.stopPropagation(); zone.classList.remove('dragging');
    }));
    zone.addEventListener('drop', e => {
        const f = e.dataTransfer.files?.[0]; if (!f) return;
        // keep the file on the input for predict()
        fileInput.files = e.dataTransfer.files;
        readPreview(f, previewImg, placeholder);
    });
    fileInput.addEventListener('change', () => {
        const f = fileInput.files?.[0]; if (!f) return;
        readPreview(f, previewImg, placeholder);
    });
}

function readPreview(file, imgEl, phEl) {
    const reader = new FileReader();
    reader.onload = () => {
        imgEl.src = reader.result;
        imgEl.hidden = false;
        if (phEl) phEl.hidden = true;
    };
    reader.readAsDataURL(file);
}

function clearPreview({ fileInput, previewImg, placeholder }) {
    // fully reset input
    fileInput.value = '';
    // fully reset preview image
    if (previewImg) {
        previewImg.src = '';
        previewImg.hidden = true;
    }
    // show placeholder text again
    if (placeholder) {
        placeholder.hidden = false;
        placeholder.textContent = 'Drag & Drop pics in or Upload';
    }
}

// Main wiring 
const el = id => document.getElementById(id);

// elements (Main)
const file      = el('file');
const dropzone  = el('dropzone');
const preview   = el('previewImg');
const ph        = el('placeholder');
const predict   = el('predictBtn');
const clearBtn  = el('clearBtn');
const predLabel = el('predLabel');

// dropzone (Main)
wireDropzone({ zone: dropzone, fileInput: file, previewImg: preview, placeholder: ph });

// clear (Main)
clearBtn.addEventListener('click', () => {
    clearPreview({ fileInput: file, previewImg: preview, placeholder: ph });
    predLabel.textContent = '–';
});

// predict (Main, ViT-only; no viz yet)
predict.addEventListener('click', async () => {
    if (!file.files?.[0]) { alert('Please upload an image first.'); return; }
    predLabel.textContent = 'ViT: (pending backend)';
});

// Extension wiring
const fileExt      = el('fileExt');
const dropzoneExt  = el('dropzoneExt');
const previewExt   = el('previewImgExt');
const phExt        = el('placeholderExt');
const predictExt   = el('predictBtnExt');
const clearBtnExt  = el('clearBtnExt');
const mathResult   = el('mathResult');

// dropzone (Extension)
wireDropzone({ zone: dropzoneExt, fileInput: fileExt, previewImg: previewExt, placeholder: phExt });

// clear (Extension)
clearBtnExt.addEventListener('click', () => {
    clearPreview({ fileInput: fileExt, previewImg: previewExt, placeholder: phExt });
    mathResult.textContent = '–';
});

// recognise (Extension, ViT as well; no viz yet)
predictExt.addEventListener('click', async () => {
    if (!fileExt.files?.[0]) { alert('Please upload an image first.'); return; }
    mathResult.textContent = 'ViT: (pending backend)';
});
