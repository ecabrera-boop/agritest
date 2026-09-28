"use strict";

(function () {
  const STORAGE_KEY = "riceguard_leaf_draft";
  const MAX_STORAGE_BYTES = 4 * 1024 * 1024;
  const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
  const ALLOWED_MIME = new Set(["image/jpeg", "image/png", "image/webp"]);
  const ALLOWED_EXT = new Set([".jpg", ".jpeg", ".png", ".webp"]);

  function loadI18n() {
    const el = document.getElementById("rg-i18n");
    if (!el) return {};
    try {
      return JSON.parse(el.textContent);
    } catch (_) {
      return {};
    }
  }

  const I18N = loadI18n();

  const UPLOAD_ERRORS = {
    invalid_format: {
      kind: "format",
      title: I18N.unsupported_format_title || "",
      message: I18N.unsupported_format_msg || "",
    },
    file_too_large: {
      kind: "size",
      title: I18N.file_too_large_title || "",
      message: I18N.file_too_large_msg || "",
    },
    not_rice_leaf: {
      kind: "leaf",
      title: I18N.not_rice_leaf_title || "",
      message: "",
    },
    server_error: {
      kind: "format",
      title: I18N.could_not_check_title || "",
      message: I18N.could_not_check_msg || "",
    },
  };

  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("id_image");
  const uploadErrorSign = document.getElementById("upload-error-sign");
  const uploadErrorTitle = document.getElementById("upload-error-title");
  const uploadErrorMessage = document.getElementById("upload-error-message");
  const uploadErrorDismiss = document.getElementById("upload-error-dismiss");
  const previewContainer = document.getElementById("preview-container");
  const previewOriginal = document.getElementById("preview-original");
  const previewProcessing = document.getElementById("preview-processing");
  const previewProcessingText = document.getElementById("preview-processing-text");
  const layer0Warning = document.getElementById("layer0-warning");
  const layer0WarningText = document.getElementById("layer0-warning-text");
  const previewFilename = document.getElementById("preview-filename");
  const fileInputRow = document.getElementById("file-input-row");
  const removePreviewBtn = document.getElementById("remove-preview-btn");
  const changeImageBtn = document.getElementById("change-image-btn");
  const analysisActions = document.getElementById("analysis-actions");
  const form = document.getElementById("upload-form");
  const submitBtn = document.getElementById("submit-btn");
  const loadingOverlay = document.getElementById("loading-overlay");
  const cameraBtn = document.getElementById("camera-btn");
  const cameraOverlay = document.getElementById("camera-overlay");
  const cameraStream = document.getElementById("camera-stream");
  const cameraControls = document.getElementById("camera-controls");
  const captureBtn = document.getElementById("capture-btn");
  const closeCameraBtn = document.getElementById("close-camera-btn");
  const canvas = document.getElementById("capture-canvas");
  const latitudeInput = document.getElementById("id_latitude");
  const longitudeInput = document.getElementById("id_longitude");
  const previewLocation = document.getElementById("preview-location");
  const previewLocationText = document.getElementById("preview-location-text");
  const previewLocationCard = document.getElementById("preview-location-card");
  const previewLocationName = document.getElementById("preview-location-name");
  const previewLocationCoords = document.getElementById("preview-location-coords");
  const previewLocationMapLink = document.getElementById("preview-location-map-link");
  const previewLocationMapWrap = document.getElementById("preview-location-map-wrap");
  const previewLocationMap = document.getElementById("preview-location-map");
  const previewLocationHint = document.getElementById("preview-location-hint");
  const useLocationBtn = document.getElementById("use-location-btn");
  const previewUseLocationBtn = document.getElementById("preview-use-location-btn");

  const validateUrl = form?.dataset?.validateUrl || "/api/validate-layer0/";
  const locateUrl = form?.dataset?.locateUrl || "/api/locate/";
  let mediaStream = null;
  let lastFile = null;
  let lastFileName = "leaf.jpg";
  let lastOriginalDataUrl = null;
  let lastCoords = null;
  let lastLocation = null;
  let locationFromExif = false;
  let locateRequestId = 0;
  let geoWatchId = null;
  let locationRequestInFlight = false;
  let layer0Passed = false;
  let validateRequestId = 0;
  let validateAbortController = null;

  if (dropZone && fileInput) {
    const openFilePickerWithLocation = () => {
      // Must run inside the same tap so mobile browsers open the GPS permission prompt.
      requestCaptureLocation({ force: true, fresh: true });
      fileInput.click();
    };

    dropZone.addEventListener("click", openFilePickerWithLocation);
    dropZone.addEventListener("pointerdown", () => {
      requestCaptureLocation({ force: true, fresh: true });
    }, { passive: true });

    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));

    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
      const file = e.dataTransfer.files[0];
      if (file) {
        setFile(file, { requestLocation: true });
      }
    });

    fileInput.addEventListener("pointerdown", () => {
      requestCaptureLocation({ force: true, fresh: true });
    }, { passive: true });
    fileInput.addEventListener("click", () => {
      requestCaptureLocation({ force: true, fresh: true });
    });
    fileInput.addEventListener("change", () => {
      if (fileInput.files[0]) {
        setFile(fileInput.files[0], { requestLocation: true });
      } else {
        clearPreview();
      }
    });
  }

  function fileExtension(name) {
    const match = (name || "").toLowerCase().match(/\.[a-z0-9]+$/);
    return match ? match[0] : "";
  }

  function validateClientFile(file) {
    if (!file) {
      return { code: "invalid_format", ...UPLOAD_ERRORS.invalid_format };
    }
    const ext = fileExtension(file.name);
    const mimeOk = file.type && ALLOWED_MIME.has(file.type);
    const extOk = ext && ALLOWED_EXT.has(ext);
    if (!mimeOk && !extOk) {
      return { code: "invalid_format", ...UPLOAD_ERRORS.invalid_format };
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      return { code: "file_too_large", ...UPLOAD_ERRORS.file_too_large };
    }
    return null;
  }

  function uploadErrorFromCode(code, messageOverride) {
    const base = UPLOAD_ERRORS[code] || UPLOAD_ERRORS.server_error;
    return {
      code,
      kind: base.kind,
      title: base.title,
      message: messageOverride || base.message,
    };
  }

  function showUploadError(error) {
    if (!uploadErrorSign || !error) return;
    const kind = error.kind || "format";
    uploadErrorSign.classList.remove(
      "upload-error-format",
      "upload-error-size",
      "upload-error-leaf",
      "d-none"
    );
    uploadErrorSign.classList.add(`upload-error-${kind}`);
    if (uploadErrorTitle) uploadErrorTitle.textContent = error.title || I18N.upload_problem || "";
    if (uploadErrorMessage) {
      uploadErrorMessage.textContent = error.message || "";
    }
    if (dropZone) dropZone.classList.add("drop-zone-error");
    uploadErrorSign.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function hideUploadError() {
    if (uploadErrorSign) {
      uploadErrorSign.classList.add("d-none");
      uploadErrorSign.classList.remove(
        "upload-error-format",
        "upload-error-size",
        "upload-error-leaf"
      );
    }
    if (uploadErrorTitle) uploadErrorTitle.textContent = "";
    if (uploadErrorMessage) uploadErrorMessage.textContent = "";
    if (dropZone) dropZone.classList.remove("drop-zone-error");
  }

  if (uploadErrorDismiss) {
    uploadErrorDismiss.addEventListener("click", () => {
      hideUploadError();
      if (!layer0Passed && previewContainer && !previewContainer.classList.contains("d-none")) {
        clearPreview();
      }
    });
  }

  function hidePreviewLocation() {
    if (previewLocation) previewLocation.classList.add("d-none");
    if (previewLocationText) previewLocationText.textContent = "";
    if (previewLocationCard) previewLocationCard.classList.add("d-none");
    if (previewLocationName) previewLocationName.textContent = "";
    if (previewLocationCoords) {
      previewLocationCoords.textContent = "";
      previewLocationCoords.classList.add("d-none");
    }
    if (previewLocationMapLink) {
      previewLocationMapLink.href = "";
      previewLocationMapLink.classList.add("d-none");
    }
    if (previewLocationMapWrap) previewLocationMapWrap.classList.add("d-none");
    if (previewLocationMap) previewLocationMap.src = "";
    if (previewLocationHint) previewLocationHint.classList.add("d-none");
  }

  function showLocationHint(show) {
    if (!previewLocationHint) return;
    previewLocationHint.classList.toggle("d-none", !show);
  }

  function showFindingLocation() {
    const label = I18N.getting_location || I18N.finding_location || "";
    if (previewLocation && previewLocationText) {
      previewLocationText.textContent = label;
      previewLocation.classList.remove("d-none");
    }
    if (previewLocationCard && previewLocationName) {
      previewLocationName.replaceChildren();
      const icon = document.createElement("i");
      icon.className = "bi bi-geo-alt text-muted me-1";
      previewLocationName.appendChild(icon);
      previewLocationName.appendChild(document.createTextNode(label));
      previewLocationCard.classList.remove("d-none");
    }
    showLocationHint(false);
    setLocationButtonsBusy(true);
  }

  function setLocationButtonsBusy(busy) {
    locationRequestInFlight = busy;
    [useLocationBtn, previewUseLocationBtn].forEach((btn) => {
      if (!btn) return;
      btn.disabled = !!busy;
      if (busy) {
        btn.dataset.busy = "1";
        btn.innerHTML =
          '<span class="spinner-border spinner-border-sm me-2" role="status"></span>' +
          (I18N.getting_location || I18N.finding_location || "");
      } else if (btn.dataset.busy === "1") {
        btn.dataset.busy = "0";
        btn.innerHTML =
          '<i class="bi bi-geo-alt-fill me-2"></i>' +
          (I18N.use_my_location || "");
      }
    });
  }

  function stopGeoWatch() {
    if (geoWatchId != null && navigator.geolocation) {
      try {
        navigator.geolocation.clearWatch(geoWatchId);
      } catch (_) {
        /* ignore */
      }
      geoWatchId = null;
    }
  }

  function markLocationFailed(label) {
    setLocationButtonsBusy(false);
    showLocationHint(true);
    if (previewLocation && previewLocationText) {
      previewLocationText.textContent = label;
      previewLocation.classList.remove("d-none");
    }
    if (previewLocationCard && previewLocationName) {
      previewLocationName.replaceChildren();
      const icon = document.createElement("i");
      icon.className = "bi bi-geo-alt text-muted me-1";
      previewLocationName.appendChild(icon);
      previewLocationName.appendChild(document.createTextNode(label));
      previewLocationCard.classList.remove("d-none");
    }
  }

  function scrollLocationIntoView() {
    if (!isMobileCameraView()) return;
    const target = previewLocationCard || previewLocation;
    if (!target || target.classList.contains("d-none")) return;
    requestAnimationFrame(() => {
      target.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  }

  function renderPreviewLocation(loc) {
    if (!loc) {
      hidePreviewLocation();
      return;
    }
    const display =
      loc.location_display ||
      loc.location_name ||
      I18N.location_captured ||
      "Capture location saved";
    const coords = loc.coordinates_display || "";
    if (previewLocation && previewLocationText) {
      previewLocationText.textContent = display;
      previewLocation.classList.remove("d-none");
    }
    if (previewLocationCard && previewLocationName) {
      previewLocationName.replaceChildren();
      const icon = document.createElement("i");
      icon.className = "bi bi-geo-alt-fill text-success me-1";
      previewLocationName.appendChild(icon);
      previewLocationName.appendChild(document.createTextNode(display));
      previewLocationCard.classList.remove("d-none");
    }
    if (previewLocationCoords) {
      if (coords && loc.location_name) {
        previewLocationCoords.textContent = coords;
        previewLocationCoords.classList.remove("d-none");
      } else {
        previewLocationCoords.textContent = "";
        previewLocationCoords.classList.add("d-none");
      }
    }
    if (previewLocationMapLink) {
      if (loc.maps_url) {
        previewLocationMapLink.href = loc.maps_url;
        previewLocationMapLink.classList.remove("d-none");
      } else {
        previewLocationMapLink.classList.add("d-none");
      }
    }
    if (previewLocationMapWrap && previewLocationMap && loc.map_embed_url) {
      if (previewLocationMap.src !== loc.map_embed_url) {
        previewLocationMap.src = loc.map_embed_url;
      }
      previewLocationMapWrap.classList.remove("d-none");
    }
    showLocationHint(false);
    scrollLocationIntoView();
    setLocationButtonsBusy(false);
  }

  function applyLocation(loc, options = {}) {
    if (!loc) return;
    const lat = Number(loc.latitude);
    const lng = Number(loc.longitude);
    if (!Number.isFinite(lat) || !Number.isFinite(lng)) return;
    const fromExif = options.fromExif === true || loc.location_source === "exif";
    if (locationFromExif && !fromExif) return;

    lastCoords = { latitude: lat, longitude: lng };
    lastLocation = { ...loc, latitude: lat, longitude: lng };
    if (fromExif) locationFromExif = true;
    if (latitudeInput) latitudeInput.value = String(lat);
    if (longitudeInput) longitudeInput.value = String(lng);
    renderPreviewLocation(lastLocation);
  }

  function setCaptureCoords(coords) {
    if (!coords || !Number.isFinite(coords.latitude) || !Number.isFinite(coords.longitude)) {
      return;
    }
    applyLocation({
      latitude: coords.latitude,
      longitude: coords.longitude,
      location_display: lastLocation?.location_display || I18N.finding_location || "",
      location_name: lastLocation?.location_name || "",
      coordinates_display: lastLocation?.coordinates_display || "",
      maps_url: lastLocation?.maps_url || "",
      map_embed_url: lastLocation?.map_embed_url || "",
      location_source: "device",
    });
    resolvePlaceName();
  }

  function clearCaptureCoords() {
    stopGeoWatch();
    lastCoords = null;
    lastLocation = null;
    locationFromExif = false;
    locateRequestId += 1;
    if (latitudeInput) latitudeInput.value = "";
    if (longitudeInput) longitudeInput.value = "";
    setLocationButtonsBusy(false);
    hidePreviewLocation();
  }

  function appendLocationFields(formData) {
    if (lastCoords) {
      formData.append("latitude", String(lastCoords.latitude));
      formData.append("longitude", String(lastCoords.longitude));
    }
  }

  function resolvePlaceName() {
    if (!lastCoords && !lastFile) return;
    if (lastLocation && lastLocation.location_name) return;
    const requestId = ++locateRequestId;
    const formData = new FormData();
    if (lastFile) formData.append("image", lastFile);
    appendLocationFields(formData);
    fetch(locateUrl, {
      method: "POST",
      headers: { "X-CSRFToken": getCsrfToken() },
      body: formData,
    })
      .then((res) => res.json().catch(() => ({})))
      .then((body) => {
        if (requestId !== locateRequestId) return;
        if (body.found && body.location) {
          applyLocation(body.location, { fromExif: body.location.location_source === "exif" });
        }
      })
      .catch(() => {});
  }

  function requestCaptureLocation(options = {}) {
    const force = options.force === true;
    const fresh = options.fresh !== false;

    if (!navigator.geolocation) {
      markLocationFailed(I18N.location_unavailable || "");
      return;
    }
    if (!window.isSecureContext && location.hostname !== "localhost" && location.hostname !== "127.0.0.1") {
      markLocationFailed(I18N.location_denied || "");
      return;
    }

    if (force) {
      // New file/camera action: always re-ask device GPS.
      stopGeoWatch();
      lastCoords = null;
      lastLocation = null;
      locationFromExif = false;
      locateRequestId += 1;
      if (latitudeInput) latitudeInput.value = "";
      if (longitudeInput) longitudeInput.value = "";
    } else if (lastCoords && !force) {
      return;
    }

    showFindingLocation();

    const applyPos = (pos) => {
      setCaptureCoords({
        latitude: pos.coords.latitude,
        longitude: pos.coords.longitude,
      });
      stopGeoWatch();
      setLocationButtonsBusy(false);
    };

    const fail = (err) => {
      if (lastCoords) {
        setLocationButtonsBusy(false);
        return;
      }
      const denied = err && err.code === 1;
      markLocationFailed(
        denied
          ? I18N.location_denied || ""
          : I18N.location_unavailable || ""
      );
    };

    const geoOptions = {
      enableHighAccuracy: true,
      timeout: isMobileCameraView() ? 20000 : 12000,
      maximumAge: fresh ? 0 : 60000,
    };

    navigator.geolocation.getCurrentPosition(
      applyPos,
      (err) => {
        if (err && err.code === 1) {
          fail(err);
          return;
        }
        // Indoor phones often fail high-accuracy; retry once, then watch briefly.
        navigator.geolocation.getCurrentPosition(
          applyPos,
          () => {
            try {
              stopGeoWatch();
              geoWatchId = navigator.geolocation.watchPosition(
                (pos) => applyPos(pos),
                fail,
                { enableHighAccuracy: false, timeout: 15000, maximumAge: 0 }
              );
              setTimeout(() => {
                if (!lastCoords) stopGeoWatch();
                if (!lastCoords) fail(err);
              }, 12000);
            } catch (_) {
              fail(err);
            }
          },
          { enableHighAccuracy: false, timeout: 12000, maximumAge: fresh ? 0 : 120000 }
        );
      },
      geoOptions
    );
  }

  function waitForCaptureLocation(ms) {
    if (lastCoords) return Promise.resolve(lastCoords);
    if (!navigator.geolocation) return Promise.resolve(null);
    requestCaptureLocation({ force: false, fresh: true });
    return new Promise((resolve) => {
      const started = Date.now();
      const timer = setInterval(() => {
        if (lastCoords || Date.now() - started >= ms) {
          clearInterval(timer);
          resolve(lastCoords);
        }
      }, 200);
    });
  }

  function setFile(file, options = {}) {
    hideUploadError();
    const clientError = validateClientFile(file);
    if (clientError) {
      showUploadError(clientError);
      if (fileInput) fileInput.value = "";
      return;
    }

    lastFile = file;
    lastFileName = file.name || "leaf.jpg";
    layer0Passed = false;
    if (options.requestLocation !== false) {
      requestCaptureLocation({ force: true, fresh: true });
    }
    assignFileInput(file);
    const reader = new FileReader();
    reader.onload = (e) => {
      lastOriginalDataUrl = e.target.result;
      applyPreview(lastOriginalDataUrl, lastFileName, file.type);
    };
    reader.readAsDataURL(file);
  }

  function getCsrfToken() {
    const input = document.querySelector("[name=csrfmiddlewaretoken]");
    return input ? input.value : "";
  }

  function setProcessingVisible(show, text) {
    if (previewProcessingText && text) previewProcessingText.textContent = text;
    if (previewProcessing) previewProcessing.classList.toggle("d-none", !show);
    if (submitBtn) submitBtn.disabled = show;
  }

  function setAnalysisReady(ready) {
    layer0Passed = ready;
    if (analysisActions) analysisActions.classList.toggle("d-none", !ready);
  }

  function showLayer0Rejection(message) {
    setAnalysisReady(false);
    showUploadError({
      code: "not_rice_leaf",
      kind: "leaf",
      title: UPLOAD_ERRORS.not_rice_leaf.title,
      message: message || I18N.not_rice_leaf_msg || "",
    });
  }

  // Non-blocking notice: the gate flagged the photo but analysis can proceed.
  function setLayer0Warning(message) {
    if (!layer0Warning) return;
    if (message && layer0WarningText) layer0WarningText.textContent = message;
    layer0Warning.classList.toggle("d-none", !message);
  }

  function cancelPendingValidation() {
    if (validateAbortController) {
      validateAbortController.abort();
      validateAbortController = null;
    }
  }

  function postImage(url, file, signal) {
    const formData = new FormData();
    formData.append("image", file);
    appendLocationFields(formData);
    return fetch(url, {
      method: "POST",
      headers: { "X-CSRFToken": getCsrfToken() },
      body: formData,
      signal,
    });
  }

  function validateLayer0(file, signal) {
    return postImage(validateUrl, file, signal).then(async (res) => {
      const body = await res.json().catch(() => ({}));
      if (body.code === "invalid_format" || body.code === "file_too_large") {
        return {
          accepted: false,
          clientCode: body.code,
          message: body.message || body.error || "",
        };
      }
      if (res.ok && body.accepted === true) {
        return { accepted: true, warning: body.warning || "", location: body.location || null };
      }
      if (!res.ok) {
        return {
          accepted: false,
          clientCode: "server_error",
          message: body.message || body.error || UPLOAD_ERRORS.server_error.message,
        };
      }
      return {
        accepted: false,
        clientCode: "not_rice_leaf",
        reason: body.reason || "",
        message: body.message || body.error || I18N.cannot_analyse || "",
      };
    });
  }

  function applyPreview(originalDataUrl, fileName, mimeType) {
    if (previewOriginal) previewOriginal.src = originalDataUrl;
    if (previewFilename) previewFilename.textContent = fileName || "leaf.jpg";
    hideUploadError();
    setLayer0Warning("");
    setAnalysisReady(false);
    showPreview();

    const requestId = ++validateRequestId;
    cancelPendingValidation();
    validateAbortController = new AbortController();
    const { signal } = validateAbortController;
    setProcessingVisible(true, I18N.checking_image || "");

    if (!lastFile) {
      lastFile = dataUrlToFile(originalDataUrl, fileName, mimeType);
      assignFileInput(lastFile);
    }

    validateLayer0(lastFile, signal)
      .then((result) => {
        if (requestId !== validateRequestId) return;

        if (!result.accepted) {
          if (result.clientCode === "invalid_format" || result.clientCode === "file_too_large") {
            showUploadError(uploadErrorFromCode(result.clientCode, result.message));
            clearPreview();
            return;
          }
          showLayer0Rejection(result.message);
          persistDraft(originalDataUrl, fileName, mimeType);
          return;
        }

        hideUploadError();
        setLayer0Warning(result.warning || "");
        setAnalysisReady(true);
        if (result.location) {
          applyLocation(result.location, { fromExif: result.location.location_source === "exif" });
        } else {
          resolvePlaceName();
        }
        persistDraft(originalDataUrl, fileName, mimeType);
      })
      .catch((err) => {
        if (err.name === "AbortError" || requestId !== validateRequestId) return;
        showLayer0Rejection(err.message || I18N.could_not_validate || "");
      })
      .finally(() => {
        if (requestId === validateRequestId) {
          validateAbortController = null;
          setProcessingVisible(false);
        }
      });
  }

  function assignFileInput(file) {
    const dt = new DataTransfer();
    dt.items.add(file);
    if (fileInput) fileInput.files = dt.files;
  }

  function showPreview() {
    if (previewContainer) previewContainer.classList.remove("d-none");
    if (fileInputRow) fileInputRow.classList.add("d-none");
    if (dropZone) dropZone.classList.add("d-none");
  }

  function clearPreview(options = {}) {
    const keepUploadError = options.keepUploadError === true;
    cancelPendingValidation();
    validateRequestId += 1;
    lastFile = null;
    lastOriginalDataUrl = null;
    layer0Passed = false;
    clearCaptureCoords();
    if (!keepUploadError) {
      hideUploadError();
    }
    setLayer0Warning("");
    setAnalysisReady(false);
    if (previewOriginal) previewOriginal.src = "";
    if (previewFilename) previewFilename.textContent = "";
    if (previewContainer) previewContainer.classList.add("d-none");
    if (fileInputRow) fileInputRow.classList.remove("d-none");
    if (dropZone) dropZone.classList.remove("d-none");
    if (fileInput) fileInput.value = "";
    setProcessingVisible(false);
    clearDraft();
  }

  if (removePreviewBtn) {
    removePreviewBtn.addEventListener("click", clearPreview);
  }

  if (changeImageBtn && fileInput) {
    changeImageBtn.addEventListener("click", () => {
      requestCaptureLocation({ force: true, fresh: true });
      fileInput.click();
    });
  }

  [useLocationBtn, previewUseLocationBtn].forEach((btn) => {
    if (!btn) return;
    btn.addEventListener("click", () => {
      requestCaptureLocation({ force: true, fresh: true });
    });
  });

  function persistDraft(originalDataUrl, fileName, mimeType) {
    const payload = JSON.stringify({
      originalDataUrl,
      fileName,
      mimeType,
      layer0Passed,
      latitude: lastCoords ? lastCoords.latitude : null,
      longitude: lastCoords ? lastCoords.longitude : null,
      location: lastLocation,
    });
    if (payload.length <= MAX_STORAGE_BYTES) {
      try {
        localStorage.setItem(STORAGE_KEY, payload);
        return;
      } catch (_) {
        /* fall through */
      }
    }
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ originalDataUrl, fileName, mimeType, layer0Passed })
      );
    } catch (_) {
      /* storage full */
    }
  }

  function clearDraft() {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (_) {
      /* ignore */
    }
  }

  function startFreshScan() {
    clearDraft();
    clearPreview();
  }

  function consumeNewScanQuery() {
    const params = new URLSearchParams(window.location.search);
    if (!params.has("new_scan")) return false;
    startFreshScan();
    params.delete("new_scan");
    const query = params.toString();
    const nextUrl = query ? `${window.location.pathname}?${query}` : window.location.pathname;
    window.history.replaceState({}, "", nextUrl);
    return true;
  }

  function dataUrlToFile(dataUrl, fileName, mimeType) {
    const parts = dataUrl.split(",");
    const mime = mimeType || (parts[0].match(/:(.*?);/) || [])[1] || "image/jpeg";
    const binary = atob(parts[1]);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return new File([bytes], fileName || "leaf.jpg", { type: mime });
  }

  function restoreDraft() {
    let draft;
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      draft = JSON.parse(raw);
    } catch (_) {
      return;
    }
    const original = draft.originalDataUrl || draft.dataUrl;
    if (!original) return;

    lastFileName = draft.fileName || "leaf.jpg";
    lastOriginalDataUrl = original;
    lastFile = dataUrlToFile(original, lastFileName, draft.mimeType);
    if (draft.location) {
      applyLocation(draft.location, { fromExif: draft.location.location_source === "exif" });
    } else if (draft.latitude != null && draft.longitude != null) {
      setCaptureCoords({ latitude: Number(draft.latitude), longitude: Number(draft.longitude) });
    } else {
      requestCaptureLocation();
    }
    assignFileInput(lastFile);
    applyPreview(original, lastFileName, draft.mimeType);
  }

  if (form) {
    form.addEventListener("submit", (event) => {
      event.preventDefault();

      if (!layer0Passed) {
        if (lastFile) {
          const clientError = validateClientFile(lastFile);
          if (clientError) {
            showUploadError(clientError);
          } else {
            showUploadError(
              uploadErrorFromCode(
                "not_rice_leaf",
                I18N.not_passed_check || ""
              )
            );
          }
        } else {
          showUploadError(
            uploadErrorFromCode("invalid_format", I18N.choose_image_first || "")
          );
        }
        return;
      }

      if (!lastFile) {
        showUploadError(
          uploadErrorFromCode("invalid_format", I18N.choose_image_first || "")
        );
        return;
      }

      const submitLabel = submitBtn ? submitBtn.innerHTML : "";
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>' + (I18N.analysing_ellipsis || "");
      }
      if (loadingOverlay) loadingOverlay.classList.remove("d-none");
      clearDraft();

      waitForCaptureLocation(isMobileCameraView() ? 8000 : 4000).then((coords) => {
        const formData = new FormData();
        formData.append("image", lastFile);
        formData.append("csrfmiddlewaretoken", getCsrfToken());
        if (coords) {
          formData.append("latitude", String(coords.latitude));
          formData.append("longitude", String(coords.longitude));
        }

        return fetch(form.action, {
          method: "POST",
          headers: { "X-Requested-With": "XMLHttpRequest" },
          body: formData,
        });
      })
        .then(async (res) => {
          const body = await res.json().catch(() => ({}));
          if (!res.ok) {
            throw new Error(body.error || body.message || I18N.upload_failed || "");
          }
          if (body.redirect_url) {
            window.location.assign(body.redirect_url);
            return;
          }
          throw new Error(I18N.no_result_url || "");
        })
        .catch((err) => {
          if (loadingOverlay) loadingOverlay.classList.add("d-none");
          if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = submitLabel;
          }
          showUploadError({
            code: "server_error",
            kind: "format",
            title: I18N.analysis_could_not_finish || "",
            message:
              err.message ||
              I18N.connection_timeout ||
              "",
          });
        });
    });
  }

  const CAMERA_JPEG_QUALITY = 0.95;
  const CAMERA_FOCUS_SETTLE_MS = 350;
  const CAMERA_CONSTRAINT_STEPS = [
    {
      facingMode: { ideal: "environment" },
      width: { ideal: 3840, min: 1280 },
      height: { ideal: 2160, min: 720 },
      frameRate: { ideal: 30, max: 30 },
    },
    {
      facingMode: { ideal: "environment" },
      width: { ideal: 1920, min: 1280 },
      height: { ideal: 1080, min: 720 },
    },
    {
      facingMode: { ideal: "environment" },
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
    { facingMode: { ideal: "environment" } },
    true,
  ];

  let imageCapture = null;
  let capturingPhoto = false;

  function isMobileCameraView() {
    return window.matchMedia("(max-width: 767.98px)").matches;
  }

  function setCaptureBusy(busy) {
    capturingPhoto = busy;
    if (!captureBtn) return;
    captureBtn.disabled = busy;
    captureBtn.classList.toggle("is-capturing", busy);
    if (busy) {
      captureBtn.dataset.prevLabel = captureBtn.innerHTML;
      captureBtn.innerHTML =
        '<span class="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span>' +
        (I18N.capturing || I18N.capture || "Capture");
    } else if (captureBtn.dataset.prevLabel) {
      captureBtn.innerHTML = captureBtn.dataset.prevLabel;
      delete captureBtn.dataset.prevLabel;
    }
  }

  function openCameraOverlay() {
    if (cameraOverlay) {
      cameraOverlay.classList.remove("d-none");
      cameraOverlay.classList.add("is-open");
      cameraOverlay.setAttribute("aria-hidden", "false");
    }
    if (cameraStream) cameraStream.classList.remove("d-none");
    if (cameraControls) cameraControls.classList.remove("d-none");
    if (cameraBtn) cameraBtn.classList.add("d-none");
    document.body.classList.add("camera-open");

    requestAnimationFrame(() => {
      if (isMobileCameraView()) {
        window.scrollTo(0, 0);
        return;
      }
      if (cameraOverlay) {
        cameraOverlay.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
      }
    });
  }

  function stopCamera() {
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
    imageCapture = null;
    capturingPhoto = false;
    if (captureBtn) {
      captureBtn.disabled = false;
      captureBtn.classList.remove("is-capturing");
      if (captureBtn.dataset.prevLabel) {
        captureBtn.innerHTML = captureBtn.dataset.prevLabel;
        delete captureBtn.dataset.prevLabel;
      }
    }
    if (cameraStream) {
      cameraStream.srcObject = null;
      cameraStream.classList.add("d-none");
    }
    if (cameraControls) cameraControls.classList.add("d-none");
    if (cameraOverlay) {
      cameraOverlay.classList.add("d-none");
      cameraOverlay.classList.remove("is-open");
      cameraOverlay.setAttribute("aria-hidden", "true");
    }
    document.body.classList.remove("camera-open");
    if (cameraBtn) cameraBtn.classList.remove("d-none");
  }

  async function openHighResCamera() {
    let lastError = null;
    for (const video of CAMERA_CONSTRAINT_STEPS) {
      try {
        return await navigator.mediaDevices.getUserMedia({ video, audio: false });
      } catch (err) {
        lastError = err;
      }
    }
    throw lastError || new Error("Camera unavailable");
  }

  function pickSupportedConstraint(capabilities, key, preferred) {
    const supported = capabilities && capabilities[key];
    if (!supported) return null;
    if (Array.isArray(supported)) {
      for (const value of preferred) {
        if (supported.includes(value)) return value;
      }
      return null;
    }
    return preferred[0] || null;
  }

  async function tuneCameraTrack(track) {
    if (!track || typeof track.getCapabilities !== "function") return;

    let capabilities = {};
    try {
      capabilities = track.getCapabilities() || {};
    } catch (_) {
      return;
    }

    const advanced = [];
    const focusMode = pickSupportedConstraint(capabilities, "focusMode", [
      "continuous",
      "single-shot",
      "auto",
    ]);
    if (focusMode) advanced.push({ focusMode });

    const exposureMode = pickSupportedConstraint(capabilities, "exposureMode", [
      "continuous",
      "auto",
    ]);
    if (exposureMode) advanced.push({ exposureMode });

    const whiteBalanceMode = pickSupportedConstraint(capabilities, "whiteBalanceMode", [
      "continuous",
      "auto",
    ]);
    if (whiteBalanceMode) advanced.push({ whiteBalanceMode });

    if (!advanced.length) return;
    try {
      await track.applyConstraints({ advanced });
    } catch (_) {
      // Browser/device may advertise modes it cannot apply.
    }

    // Nudge single-shot autofocus toward the frame center when available.
    if (focusMode === "single-shot" && typeof track.applyConstraints === "function") {
      try {
        await track.applyConstraints({
          advanced: [{ focusMode: "single-shot", pointsOfInterest: [{ x: 0.5, y: 0.5 }] }],
        });
      } catch (_) {
        /* ignore */
      }
    }
  }

  function waitForVideoFrame(video, timeoutMs = 4000) {
    return new Promise((resolve, reject) => {
      if (!video) {
        reject(new Error("No video element"));
        return;
      }
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        resolve();
        return;
      }
      const timer = setTimeout(() => {
        cleanup();
        reject(new Error("Camera preview timed out"));
      }, timeoutMs);
      const onReady = () => {
        if (video.videoWidth > 0 && video.videoHeight > 0) {
          cleanup();
          resolve();
        }
      };
      const cleanup = () => {
        clearTimeout(timer);
        video.removeEventListener("loadedmetadata", onReady);
        video.removeEventListener("loadeddata", onReady);
        video.removeEventListener("resize", onReady);
      };
      video.addEventListener("loadedmetadata", onReady);
      video.addEventListener("loadeddata", onReady);
      video.addEventListener("resize", onReady);
    });
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function blobToJpegFile(blob, name) {
    if (!blob) return null;
    if (blob.type === "image/jpeg") {
      return new File([blob], name, { type: "image/jpeg" });
    }
    return null;
  }

  function canvasToJpegBlob(sourceCanvas, quality) {
    return new Promise((resolve) => {
      sourceCanvas.toBlob((blob) => resolve(blob), "image/jpeg", quality);
    });
  }

  async function compressImageBlob(blob, maxBytes) {
    if (!blob || blob.size <= maxBytes) return blob;
    const bitmap = await createImageBitmap(blob);
    const shrinkSteps = [1, 0.85, 0.7, 0.55];
    const qualities = [CAMERA_JPEG_QUALITY, 0.9, 0.84, 0.78];
    let best = blob;
    try {
      for (const scale of shrinkSteps) {
        const w = Math.max(1, Math.round(bitmap.width * scale));
        const h = Math.max(1, Math.round(bitmap.height * scale));
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, w, h);
        ctx.drawImage(bitmap, 0, 0, w, h);
        for (const quality of qualities) {
          const next = await canvasToJpegBlob(canvas, quality);
          if (!next) continue;
          best = next;
          if (next.size <= maxBytes) return next;
        }
      }
    } finally {
      if (typeof bitmap.close === "function") bitmap.close();
    }
    return best;
  }

  async function captureFromImageCapture(track) {
    if (!imageCapture || typeof imageCapture.takePhoto !== "function") return null;
    try {
      let photoSettings = undefined;
      if (typeof imageCapture.getPhotoCapabilities === "function") {
        const caps = await imageCapture.getPhotoCapabilities().catch(() => null);
        if (caps && caps.imageWidth && caps.imageHeight) {
          photoSettings = {
            imageWidth: caps.imageWidth.max || caps.imageWidth,
            imageHeight: caps.imageHeight.max || caps.imageHeight,
          };
        }
      }
      const blob = photoSettings
        ? await imageCapture.takePhoto(photoSettings)
        : await imageCapture.takePhoto();
      return blob || null;
    } catch (_) {
      // Fall back to video-frame grab below.
      return null;
    }
  }

  async function captureFromVideoFrame() {
    if (!cameraStream || !cameraStream.videoWidth || !cameraStream.videoHeight) {
      throw new Error("Camera frame not ready");
    }
    canvas.width = cameraStream.videoWidth;
    canvas.height = cameraStream.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(cameraStream, 0, 0, canvas.width, canvas.height);
    const blob = await canvasToJpegBlob(canvas, CAMERA_JPEG_QUALITY);
    if (!blob) throw new Error("Could not encode photo");
    return blob;
  }

  async function finalizeCapturedBlob(blob) {
    let photo = blob;
    if (!photo || photo.type !== "image/jpeg") {
      // Convert non-JPEG stills (e.g. some ImageCapture outputs) via canvas.
      const bitmap = await createImageBitmap(photo);
      try {
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        canvas.getContext("2d").drawImage(bitmap, 0, 0);
        photo = await canvasToJpegBlob(canvas, CAMERA_JPEG_QUALITY);
      } finally {
        if (typeof bitmap.close === "function") bitmap.close();
      }
    }
    photo = await compressImageBlob(photo, MAX_UPLOAD_BYTES);
    const file =
      blobToJpegFile(photo, "camera-capture.jpg") ||
      new File([photo], "camera-capture.jpg", { type: "image/jpeg" });
    setFile(file, { requestLocation: true });
    stopCamera();
    if (isMobileCameraView()) {
      setTimeout(() => {
        if (previewContainer) {
          previewContainer.scrollIntoView({ behavior: "smooth", block: "start" });
        }
        scrollLocationIntoView();
      }, 120);
    }
  }

  if (cameraBtn) {
    cameraBtn.addEventListener("click", async () => {
      requestCaptureLocation({ force: true, fresh: true });
      try {
        mediaStream = await openHighResCamera();
        const track = mediaStream.getVideoTracks()[0] || null;
        await tuneCameraTrack(track);
        imageCapture =
          track && typeof ImageCapture === "function" ? new ImageCapture(track) : null;

        cameraStream.srcObject = mediaStream;
        cameraStream.muted = true;
        cameraStream.playsInline = true;
        cameraStream.setAttribute("playsinline", "");
        cameraStream.setAttribute("webkit-playsinline", "");
        await cameraStream.play().catch(() => {});
        await waitForVideoFrame(cameraStream);
        openCameraOverlay();
        // Ask again after camera starts — GPS often settles after camera permission.
        requestCaptureLocation({ force: true, fresh: true });
      } catch (err) {
        stopCamera();
        alert((I18N.camera_not_available || "") + err.message);
      }
    });
  }

  if (captureBtn) {
    captureBtn.addEventListener("click", async () => {
      if (capturingPhoto || !mediaStream) return;
      requestCaptureLocation({ force: true, fresh: true });
      setCaptureBusy(true);
      try {
        const track = mediaStream.getVideoTracks()[0] || null;
        await tuneCameraTrack(track);
        await sleep(CAMERA_FOCUS_SETTLE_MS);
        let blob = await captureFromImageCapture(track);
        if (!blob) blob = await captureFromVideoFrame();
        await finalizeCapturedBlob(blob);
      } catch (err) {
        setCaptureBusy(false);
        alert((I18N.camera_not_available || "") + (err && err.message ? err.message : ""));
      }
    });
  }

  if (closeCameraBtn) {
    closeCameraBtn.addEventListener("click", stopCamera);
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && mediaStream) {
      stopCamera();
    }
  });

  document.querySelectorAll("[data-clear-draft]").forEach((el) => {
    el.addEventListener("click", () => clearDraft());
  });

  if (!consumeNewScanQuery()) {
    restoreDraft();
  }
})();
