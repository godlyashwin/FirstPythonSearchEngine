window.onload = function () {
    console.log("Script loaded: Starting progress polling");

    // Get query from window.searchQuery with fallback
    const query = window.searchQuery || "";
    console.log("Query is: " + query);

    // Track current displayed progress for interpolation
    let currentProgress = 0;
    const progressFill = document.getElementById("progress-fill");
    progressFill.style.width = "0%"; // Initialize progress bar

    // Function to smoothly update progress bar
    function updateProgressBar(targetProgress, duration) {
        const startProgress = currentProgress;
        const startTime = performance.now();
        const step = function (currentTime) {
            const elapsed = currentTime - startTime;
            const fraction = Math.min(elapsed / duration, 1);
            currentProgress = startProgress + (targetProgress - startProgress) * fraction;
            progressFill.style.width = currentProgress + "%";
            if (fraction < 1) {
                requestAnimationFrame(step);
            }
        };
        requestAnimationFrame(step);
    }

    // Function to poll progress status
    let retryCount = 0;
    const maxRetries = 5;
    function pollProgress() {
        if (retryCount >= maxRetries) {
            document.getElementById("progress-message").textContent = "Failed to fetch progress after multiple attempts.";
            return;
        }
        fetch("/progress_status")
            .then(function (response) {
                console.log("Progress status response status: " + response.status);
                if (!response.ok) {
                    throw new Error("HTTP error, status = " + response.status);
                }
                retryCount = 0; // Reset on success
                return response.json();
            })
            .then(function (data) {
                console.log("Progress update: ", data);
                // Update progress message
                const messageElement = document.getElementById("progress-message");
                messageElement.textContent = data.status || "Processing...";
                messageElement.setAttribute("aria-live", "polite");
                // Smoothly update progress bar
                updateProgressBar(data.progress || 0, 400);

                // If completed, fetch and display results
                if (data.completed) {
                    console.log("Search completed, fetching results");
                    fetch("/results")
                        .then(function (response) {
                            console.log("Results response status: " + response.status);
                            if (!response.ok) {
                                throw new Error("HTTP error, status = " + response.status);
                            }
                            return response.json();
                        })
                        .then(function (data) {
                            console.log("Results received: ", data);
                            // Hide spinner, progress message, and progress bar
                            document.getElementById("spinner").style.display = "none";
                            document.getElementById("progress-message").style.display = "none";
                            document.getElementById("progress-bar").style.display = "none";
                            // Show results container
                            const resultsContainer = document.getElementById("results-container");
                            resultsContainer.style.display = "block";
                            // Populate results list
                            const resultsList = document.getElementById("results-list");
                            resultsList.setAttribute("role", "list");
                            if (data.results && data.results.length > 0) {
                                data.results.forEach(function (result, index) {
                                    const li = document.createElement("li");
                                    const resultText = document.createTextNode(result + " ");
                                    const sourceLink = document.createElement("a");
                                    sourceLink.href = data.sources && data.sources[index] ? data.sources[index] : "#";
                                    const firstSlash = data.sources && data.sources[index] ? data.sources[index].indexOf("/") : -1;
                                    const secondSlash = firstSlash !== -1 ? data.sources[index].indexOf("/", firstSlash + 1) : -1;
                                    const thirdSlash = secondSlash !== -1 ? data.sources[index].indexOf("/", secondSlash + 1) : -1;
                                    sourceLink.textContent = thirdSlash !== -1 ? "[" + data.sources[index].substring(0, thirdSlash) + "]" : "[Source]";
                                    sourceLink.className = "source-link";
                                    li.appendChild(resultText);
                                    li.appendChild(sourceLink);
                                    resultsList.appendChild(li);
                                });
                            } else {
                                const li = document.createElement("li");
                                li.textContent = "No results found for your query.";
                                resultsList.appendChild(li);
                            }
                        })
                        .catch(function (error) {
                            console.error("Error fetching results: ", error);
                            document.getElementById("progress-message").textContent = "Something went wrong. Please try again later.";
                        });
                } else {
                    // Continue polling every 500ms
                    setTimeout(pollProgress, 500);
                }
            })
            .catch(function (error) {
                retryCount++;
                console.error("Error fetching progress: ", error);
                document.getElementById("progress-message").textContent = "Something went wrong. Please try again later.";
                setTimeout(pollProgress, 500);
            });
    }

    // Start polling
    pollProgress();
};