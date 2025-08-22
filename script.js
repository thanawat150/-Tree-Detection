<script>
        let originalImageData = null;
        let processedImageData = null;

        document.getElementById('gliThreshold').addEventListener('input', function () {
            document.getElementById('gliInput').value = this.value;
            refreshGLIFromUI();
        });
        document.getElementById('gliInput').addEventListener('input', function () {
            const value = Math.max(0, Math.min(0.3, parseFloat(this.value) || 0));
            this.value = value;
            document.getElementById('gliThreshold').value = value;
            refreshGLIFromUI();
        });

        document.getElementById('minArea').addEventListener('input', function () {
            document.getElementById('areaInput').value = this.value;
        });

        document.getElementById('areaInput').addEventListener('input', function () {
            const value = Math.max(0.01, parseFloat(this.value) || 0.01);
            this.value = value;
            document.getElementById('minArea').value = value;
        });

        document.getElementById('clusterDistance').addEventListener('input', function () {
            document.getElementById('clusterInput').value = this.value;
        });

        document.getElementById('clusterInput').addEventListener('input', function () {
            const value = Math.max(0, parseFloat(this.value) || 0);
            this.value = value;
            document.getElementById('clusterDistance').value = value;
        });

        document.getElementById('imageInput').addEventListener('change', function (e) {
            const file = e.target.files[0];
            if (file) {
                if (file.size > 10 * 1024 * 1024) {
                    alert('ขนาดไฟล์ใหญ่เกินไป กรุณาเลือกไฟล์ที่มีขนาดไม่เกิน 10MB');
                    return;
                }

                const reader = new FileReader();
                reader.onload = function (e) {
                    const img = new Image();
                    img.onload = function () {
                        loadImage(img);
                    };
                    img.src = e.target.result;
                };
                reader.readAsDataURL(file);
            }
        });

        function loadImage(img) {
            const canvas = document.getElementById('originalCanvas');
            const ctx = canvas.getContext('2d');

            const maxSize = 500;
            let { width, height } = img;
            const originalWidth = img.naturalWidth;
            const originalHeight = img.naturalHeight;
            const ratio = Math.min(maxSize / originalWidth, maxSize / originalHeight);
            width = originalWidth * ratio;
            height = originalHeight * ratio;

            canvas.width = width;
            canvas.height = height;
            ctx.drawImage(img, 0, 0, width, height);

            originalImageData = ctx.getImageData(0, 0, originalWidth, originalHeight);

            document.getElementById('processBtn').disabled = false;
            document.getElementById('imageSize').textContent = `${originalWidth} × ${originalHeight}`;

            refreshGLIFromUI();
        }

        function drawGLI(imageData, threshold = null) {
            const { data, width, height } = imageData;
            const out = new ImageData(width, height);
            const displayCanvas = document.getElementById('gliCanvas');
            const displayCtx = displayCanvas.getContext('2d');

            const maxSize = 500;
            const ratio = Math.min(maxSize / width, maxSize / height);
            const displayWidth = width * ratio;
            const displayHeight = height * ratio;
            displayCanvas.width = displayWidth;
            displayCanvas.height = displayHeight;

            for (let i = 0; i < data.length; i += 4) {
                const r = data[i], g = data[i + 1], b = data[i + 2];
                const gli = calculateGLI(r, g, b);

                let R = 0, G = 0, B = 0;
                if (gli >= 0) {
                    const t = Math.min(1, gli / 0.3);
                    G = 128 + Math.round(127 * t);
                    R = Math.round(128 * (1 - t));
                    B = Math.round(128 * (1 - t));
                } else {
                    const t = Math.min(1, (-gli) / 0.3);
                    R = 128 + Math.round(127 * t);
                    B = 128 + Math.round(127 * t);
                    G = Math.round(128 * (1 - t));
                }
                if (threshold !== null && gli > threshold) {
                    R = Math.min(255, R + 40);
                    G = Math.min(255, G + 40);
                    B = Math.min(255, B + 40);
                }
                out.data[i] = R;
                out.data[i + 1] = G;
                out.data[i + 2] = B;
                out.data[i + 3] = 255;
            }

            const tempCanvas = document.createElement('canvas');
            const tempCtx = tempCanvas.getContext('2d');
            
            tempCanvas.width = maxSize;
            tempCanvas.height = maxSize;
            tempCtx.putImageData(out, 0, 0);

            displayCtx.drawImage(tempCanvas, 0, 0, displayWidth, displayHeight);
        }

        function refreshGLIFromUI() {
            if (!originalImageData) return;
            const th = parseFloat(document.getElementById('gliThreshold').value);
            drawGLI(originalImageData, th);
        }

        function calculateGLI(r, g, b) {
            const epsilon = 1e-6;
            return (2 * g - r - b) / (2 * g + r + b + epsilon);
        }

        function createBinaryMask(imageData, threshold) {
            const { data, width, height } = imageData;
            const mask = new Uint8Array(width * height);
            for (let i = 0; i < data.length; i += 4) {
                const r = data[i], g = data[i + 1], b = data[i + 2];
                const gli = calculateGLI(r, g, b);
                const pixelIndex = i / 4;
                mask[pixelIndex] = gli > threshold ? 255 : 0;
            }
            return mask;
        }

        function morphologyOpen(mask, width, height) {
            const result = new Uint8Array(mask.length);
            for (let y = 1; y < height - 1; y++) {
                for (let x = 1; x < width - 1; x++) {
                    const idx = y * width + x;
                    let minVal = 255;
                    for (let dy = -1; dy <= 1; dy++) {
                        for (let dx = -1; dx <= 1; dx++) {
                            minVal = Math.min(minVal, mask[(y + dy) * width + (x + dx)]);
                        }
                    }
                    result[idx] = minVal;
                }
            }
            const dilated = new Uint8Array(mask.length);
            for (let y = 1; y < height - 1; y++) {
                for (let x = 1; x < width - 1; x++) {
                    const idx = y * width + x;
                    let maxVal = 0;
                    for (let dy = -1; dy <= 1; dy++) {
                        for (let dx = -1; dx <= 1; dx++) {
                            maxVal = Math.max(maxVal, result[(y + dy) * width + (x + dx)]);
                        }
                    }
                    dilated[idx] = maxVal;
                }
            }
            return dilated;
        }

        function findContours(mask, width, height, minArea) {
            const visited = new Set();
            const contours = [];
            function floodFill(startX, startY) {
                const stack = [[startX, startY]];
                const points = [];
                while (stack.length > 0) {
                    const [x, y] = stack.pop();
                    const idx = y * width + x;
                    if (x < 0 || x >= width || y < 0 || y >= height || visited.has(idx) || mask[idx] === 0) {
                        continue;
                    }
                    visited.add(idx);
                    points.push([x, y]);
                    stack.push([x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]);
                }
                return points;
            }
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const idx = y * width + x;
                    if (mask[idx] === 255 && !visited.has(idx)) {
                        const contour = floodFill(x, y);
                        if (contour.length >= minArea) {
                            contours.push(contour);
                        }
                    }
                }
            }
            return contours;
        }

        function getContourCenter(contour) {
            let sumX = 0, sumY = 0;
            for (const [x, y] of contour) {
                sumX += x;
                sumY += y;
            }
            return [Math.round(sumX / contour.length), Math.round(sumY / contour.length)];
        }

        function clusterCenters(centers, maxDistance) {
            const clusters = [];
            const used = new Set();
            for (let i = 0; i < centers.length; i++) {
                if (used.has(i)) continue;
                const cluster = [centers[i]];
                used.add(i);
                for (let j = i + 1; j < centers.length; j++) {
                    if (used.has(j)) continue;
                    const dist = Math.sqrt(Math.pow(centers[i][0] - centers[j][0], 2) + Math.pow(centers[i][1] - centers[j][1], 2));
                    if (dist <= maxDistance) {
                        cluster.push(centers[j]);
                        used.add(j);
                    }
                }
                let sumX = 0, sumY = 0;
                for (const [x, y] of cluster) {
                    sumX += x;
                    sumY += y;
                }
                clusters.push([Math.round(sumX / cluster.length), Math.round(sumY / cluster.length)]);
            }
            return clusters;
        }

        function calculateAndDisplayHealth() {
                const detectedTrees = parseInt(document.getElementById('treeCountInput').value) || 0;
                const deadTrees = parseInt(document.getElementById('deadTreeCount').value) || 0;
                const totalTrees = detectedTrees + deadTrees; // รวมต้นไม้ทั้งหมด
                const areaRai = parseFloat(document.getElementById('areaRaiResult').textContent) || 1;

                let percentage = 0;
                if (totalTrees > 0) {
                    percentage = (detectedTrees / totalTrees) * 100;
                }

                // คำนวณความหนาแน่นใหม่โดยใช้จำนวนต้นไม้รวมทั้งหมด
                const densityPerRai = (totalTrees / areaRai).toFixed(2);

                document.getElementById('healthPercentage').textContent = `${percentage.toFixed(2)} %`;
                document.getElementById('treeDensityRai').textContent = `${densityPerRai} ต้น/ไร่`; // อัปเดตค่าความหนาแน่น
            }
        document.getElementById('treeCountInput').addEventListener('input', calculateAndDisplayHealth);
        document.getElementById('deadTreeCount').addEventListener('input', calculateAndDisplayHealth);

        function processImage() {
            if (!originalImageData) return;

            const startTime = performance.now();
            document.getElementById('processing').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('imageContainer').style.display = 'none';

            setTimeout(() => {
                try {
                    const gliThreshold = parseFloat(document.getElementById('gliThreshold').value);
                    const minArea = parseInt(document.getElementById('minArea').value);
                    const clusterDistance = parseInt(document.getElementById('clusterDistance').value);
                    const areaSqm = parseFloat(document.getElementById('areaSqm').value);

                    if (isNaN(areaSqm) || areaSqm <= 0) {
                        alert('กรุณากรอกเนื้อที่ (ตารางเมตร) ที่ถูกต้อง');
                        document.getElementById('processing').style.display = 'none';
                        return;
                    }

                    const areaRai = areaSqm / 1600;

                    const mask = createBinaryMask(originalImageData, gliThreshold);
                    const cleanMask = morphologyOpen(mask, originalImageData.width, originalImageData.height);
                    const contours = findContours(cleanMask, originalImageData.width, originalImageData.height, minArea);
                    const centers = contours.map(contour => getContourCenter(contour));
                    const filteredCenters = clusterCenters(centers, clusterDistance);

                    drawResults(filteredCenters);

                    const processingTime = ((performance.now() - startTime) / 1000).toFixed(2);

                    document.getElementById('processing').style.display = 'none';
                    document.getElementById('imageContainer').style.display = 'grid';
                    document.getElementById('results').style.display = 'block';
                    document.getElementById('treeCountInput').value = filteredCenters.length;
                    document.getElementById('processTime').textContent = `${processingTime}s`;

                    const treeCount = filteredCenters.length;
                    const densityPerRai = (treeCount / areaRai).toFixed(2);

                    document.getElementById('areaRaiResult').textContent = areaRai.toFixed(2);
                    document.getElementById('treeDensityRai').textContent = `${densityPerRai} ต้น/ไร่`;

                    calculateAndDisplayHealth();

                } catch (error) {
                    console.error('Processing error:', error);
                    document.getElementById('processing').style.display = 'none';
                    alert('เกิดข้อผิดพลาดในการประมวลผล');
                }
            }, 100);
        }

        function drawResults(centers) {
            const canvas = document.getElementById('resultCanvas');
            const ctx = canvas.getContext('2d');
            const maxSize = 500;

            const originalWidth = maxSize;
            const originalHeight = maxSize;
            const ratio = Math.min(maxSize / originalWidth, maxSize / originalHeight);
            const displayWidth = originalWidth * ratio;
            const displayHeight = originalHeight * ratio;

            canvas.width = displayWidth;
            canvas.height = displayHeight;

            const tempCanvas = document.createElement('canvas');
            const tempCtx = tempCanvas.getContext('2d');
            tempCanvas.width = originalWidth;
            tempCanvas.height = originalHeight;
            tempCtx.putImageData(originalImageData, 0, 0);

            ctx.drawImage(tempCanvas, 0, 0, displayWidth, displayHeight);

            ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.lineWidth = 2;

            for (const [x, y] of centers) {
                const displayX = x * ratio;
                const displayY = y * ratio;
                ctx.beginPath();
                ctx.arc(displayX, displayY, 8, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();

                ctx.fillStyle = 'white';
                ctx.font = '12px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('🌳', displayX, displayY + 4);
                ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
            }
        }
    </script>
