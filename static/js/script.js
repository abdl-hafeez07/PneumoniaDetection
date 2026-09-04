const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");

const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");
const fileName = document.getElementById("fileName");

const loadingContainer = document.getElementById("loadingContainer");
const resultCard = document.getElementById("resultCard");

let selectedFile = null;


/* ================= NAVIGATION ================= */

function toggleMenu() {
    document.getElementById("navMenu").classList.toggle("active");
}

function scrollToUpload() {
    document.getElementById("analyze").scrollIntoView({
        behavior: "smooth"
    });
}


/* ================= FILE SELECTION ================= */

fileInput.addEventListener("change", function () {

    if (this.files.length > 0) {
        handleFile(this.files[0]);
    }

});


/* ================= DRAG & DROP ================= */

dropZone.addEventListener("dragover", function (event) {
    event.preventDefault();
    dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", function () {
    dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", function (event) {

    event.preventDefault();

    dropZone.classList.remove("dragging");

    const files = event.dataTransfer.files;

    if (files.length > 0) {
        handleFile(files[0]);
    }

});


/* ================= HANDLE FILE ================= */

function handleFile(file) {

    const allowedTypes = [
        "image/jpeg",
        "image/png"
    ];

    if (!allowedTypes.includes(file.type)) {
        alert("Please upload a JPG, JPEG, or PNG image.");
        return;
    }

    selectedFile = file;

    const reader = new FileReader();

    reader.onload = function (event) {

        previewImage.src = event.target.result;

        fileName.textContent = file.name;

        dropZone.style.display = "none";
        previewContainer.style.display = "block";

        loadingContainer.style.display = "none";
        resultCard.style.display = "none";

    };

    reader.readAsDataURL(file);
}


/* ================= REMOVE IMAGE ================= */

function removeImage() {

    selectedFile = null;

    fileInput.value = "";

    previewImage.src = "";

    fileName.textContent = "";

    previewContainer.style.display = "none";
    loadingContainer.style.display = "none";

    dropZone.style.display = "flex";

    resultCard.style.display = "none";
}


/* ================= ANALYZE ================= */

async function analyzeXray() {

    if (!selectedFile) {
        alert("Please select an X-ray image first.");
        return;
    }

    previewContainer.style.display = "none";
    loadingContainer.style.display = "block";
    resultCard.style.display = "none";

    try {

        const formData = new FormData();

        formData.append("file", selectedFile);

        /*
         * This connects to the Flask/FastAPI backend.
         *
         * Backend endpoint:
         * POST /predict
         */

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            let errorMsg = "Prediction request failed.";
            try {
                const errData = await response.json();
                if (errData && errData.error) {
                    errorMsg = errData.error;
                }
            } catch (_) {}
            throw new Error(errorMsg);
        }

        const data = await response.json();

        showResult(data);

    } catch (error) {

        console.error(error);

        alert(
            error.message || "Unable to analyze the image. Please make sure the prediction server is running."
        );

        previewContainer.style.display = "block";

    } finally {

        loadingContainer.style.display = "none";
    }
}


/* ================= SHOW RESULT ================= */

function showResult(data) {

    const prediction =
        String(data.predicted_class || "").toUpperCase();

    const confidence = Number(data.confidence);

    const normalProbability =
        Number(data.normal_probability);

    const pneumoniaProbability =
        Number(data.pneumonia_probability);


    document.getElementById("prediction").textContent =
        prediction || "UNKNOWN";

    document.getElementById("confidence").textContent =
        `${confidence.toFixed(2)}%`;

    document.getElementById("normalProbability").textContent =
        `${normalProbability.toFixed(2)}%`;

    document.getElementById("pneumoniaProbability").textContent =
        `${pneumoniaProbability.toFixed(2)}%`;


    document.getElementById("normalProgress").style.width =
        `${normalProbability}%`;

    document.getElementById("pneumoniaProgress").style.width =
        `${pneumoniaProbability}%`;


    const predictionBox =
        document.getElementById("predictionBox");

    if (prediction === "PNEUMONIA") {

        predictionBox.style.border =
            "1px solid #fecaca";

        predictionBox.style.background =
            "#fff7f7";

        document.getElementById("prediction").style.color =
            "#dc2626";

    } else {

        predictionBox.style.border =
            "1px solid #bbf7d0";

        predictionBox.style.background =
            "#f0fdf4";

        document.getElementById("prediction").style.color =
            "#16a34a";
    }


    resultCard.style.display = "block";

    resultCard.scrollIntoView({
        behavior: "smooth",
        block: "center"
    });
}


/* ================= RESET ================= */

function resetAnalysis() {

    selectedFile = null;

    fileInput.value = "";

    previewImage.src = "";

    fileName.textContent = "";

    document.getElementById("prediction").textContent = "—";
    document.getElementById("confidence").textContent = "—";

    document.getElementById("normalProbability").textContent = "—";
    document.getElementById("pneumoniaProbability").textContent = "—";

    document.getElementById("normalProgress").style.width = "0%";
    document.getElementById("pneumoniaProgress").style.width = "0%";

    previewContainer.style.display = "none";
    loadingContainer.style.display = "none";
    resultCard.style.display = "none";

    dropZone.style.display = "flex";

    scrollToUpload();
}