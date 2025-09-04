// 创建修改后的前端JavaScript代码
document.addEventListener('DOMContentLoaded', function() {
    // 系统启动时间
    const startTime = new Date();
    
    // 配置参数
    let MAX_DATA_POINTS = 100;
    let dataPointCount = 0;
    let isRunning = true;
    let selectedSubcarrier = 0;
    let selectedRxAntenna = 0;
    let selectedTxAntenna = 0;
    let displayMode = 'amplitude';
    
    // 获取DOM元素
    const subcarrierSelect = document.getElementById('subcarrierSelect');
    const rxAntennaSelect = document.getElementById('rxAntennaSelect');
    const txAntennaSelect = document.getElementById('txAntennaSelect');
    const displayModeSelect = document.getElementById('displayMode');
    const startButton = document.getElementById('startButton');
    const pauseButton = document.getElementById('pauseButton');
    const clearButton = document.getElementById('clearButton');
    const realTimeCheck = document.getElementById('realTimeCheck');
    const dataWindowSizeSelect = document.getElementById('dataWindowSize');
    
    // 状态显示元素
    const packetRateElement = document.getElementById('packetRate');
    const queueLengthElement = document.getElementById('queueLength');
    const processedCountElement = document.getElementById('processedCount');
    const processingRateElement = document.getElementById('processingRate');
    const uptimeElement = document.getElementById('uptime');
    const connectionStatusElement = document.getElementById('connectionStatus');
    const connectionTextElement = document.getElementById('connectionText');
    // 创建可能不存在的元素的引用变量，避免报错
    const currentSubcarrierElement = document.getElementById('currentSubcarrier') || { textContent: '' };
    const currentAntennasElement = document.getElementById('currentAntennas') || { textContent: '' };
    const latestAmplitudeElement = document.getElementById('latestAmplitude') || { textContent: '' };
    const latestPhaseElement = document.getElementById('latestPhase') || { textContent: '' };
    const dataPointCountElement = document.getElementById('dataPointCount') || { textContent: '' };
    
    // 创建图表上下文
    const amplitudeCtx = document.getElementById('amplitudeChart').getContext('2d');
    const phaseCtx = document.getElementById('phaseChart').getContext('2d');
    const spectrumCtx = document.getElementById('spectrumChart').getContext('2d');
    const packetRateCtx = document.getElementById('packetRateChart').getContext('2d');
    const queueLengthCtx = document.getElementById('queueLengthChart').getContext('2d');
    
    // 数据存储
    const amplitudeData = {
        labels: Array(MAX_DATA_POINTS).fill(''),
        datasets: [{
            label: '信号幅值',
            data: Array(MAX_DATA_POINTS).fill(null),
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.1)',
            borderWidth: 2,
            pointRadius: 1,
            tension: 0.4,
            fill: true
        }]
    };
    
    const phaseData = {
        labels: Array(MAX_DATA_POINTS).fill(''),
        datasets: [{
            label: '信号相位',
            data: Array(MAX_DATA_POINTS).fill(null),
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.1)',
            borderWidth: 2,
            pointRadius: 1,
            tension: 0.4,
            fill: true
        }]
    };
    
    // SNR数据已移除
    
    const spectrumData = {
        labels: Array.from({length: 114}, (_, i) => i),
        datasets: [{
            label: '频率响应',
            data: Array(114).fill(0),
            borderColor: 'rgb(153, 102, 255)',
            backgroundColor: 'rgba(153, 102, 255, 0.3)',
            borderWidth: 1
        }]
    };
    
    const packetRateData = {
        labels: Array(60).fill(''),
        datasets: [{
            label: '数据包速率 (包/秒)',
            data: Array(60).fill(null),
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.1)',
            borderWidth: 2,
            tension: 0.4,
            fill: true
        }]
    };
    
    const queueLengthData = {
        labels: Array(60).fill(''),
        datasets: [{
            label: '队列长度',
            data: Array(60).fill(null),
            borderColor: 'rgb(255, 159, 64)',
            backgroundColor: 'rgba(255, 159, 64, 0.1)',
            borderWidth: 2,
            tension: 0.4,
            fill: true
        }]
    };
    
    // 图表配置
    const lineChartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                display: true,
                title: {
                    display: true,
                    text: '时间点'
                }
            },
            y: {
                display: true,
                beginAtZero: true
            }
        },
        animation: {
            duration: 0
        },
        plugins: {
            legend: {
                position: 'top',
            }
        }
    };
    
    const barChartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                display: true,
                title: {
                    display: true,
                    text: '子载波'
                }
            },
            y: {
                display: true,
                beginAtZero: true
            }
        },
        animation: {
            duration: 0
        },
        plugins: {
            legend: {
                position: 'top',
            }
        }
    };
    
    // 创建图表实例
    const amplitudeChart = new Chart(amplitudeCtx, {
        type: 'line',
        data: amplitudeData,
        options: {...lineChartOptions, 
            scales: {
                ...lineChartOptions.scales, 
                y: {
                    ...lineChartOptions.scales.y, 
                    title: {display: true, text: '幅值'},
                    min: 0,
                    max: 200
                }
            }
        }
    });
    
    const phaseChart = new Chart(phaseCtx, {
        type: 'line',
        data: phaseData,
        options: {...lineChartOptions, 
            scales: {
                ...lineChartOptions.scales, 
                y: {
                    ...lineChartOptions.scales.y, 
                    title: {display: true, text: '相位(弧度)'},
                    min: -Math.PI,
                    max: Math.PI
                }
            }
        }
    });
    
    // SNR图表已移除
    
    const spectrumChart = new Chart(spectrumCtx, {
        type: 'bar',
        data: spectrumData,
        options: {...barChartOptions,
            scales: {
                ...barChartOptions.scales,
                y: {
                    ...barChartOptions.scales.y,
                    min: 0,
                    max: 100
                }
            }
        }
    });
    
    const packetRateChart = new Chart(packetRateCtx, {
        type: 'line',
        data: packetRateData,
        options: {...lineChartOptions, 
            scales: {
                ...lineChartOptions.scales, 
                y: {
                    ...lineChartOptions.scales.y, 
                    title: {display: true, text: '包/秒'},
                    min: 0,
                    max: 500
                }
            }
        }
    });
    
    const queueLengthChart = new Chart(queueLengthCtx, {
        type: 'line',
        data: queueLengthData,
        options: {...lineChartOptions, 
            scales: {
                ...lineChartOptions.scales, 
                y: {
                    ...lineChartOptions.scales.y, 
                    title: {display: true, text: '队列长度'},
                    min: 0,
                    max: 1000
                }
            }
        }
    });
    
    // 连接Socket.IO
    const socket = io();
    
    socket.on('connect', function() {
        connectionStatusElement.className = 'status-indicator status-active';
        connectionTextElement.textContent = '已连接';
        
        // 连接后立即获取分类历史
        fetchClassificationHistory();
    });
    
    socket.on('disconnect', function() {
        connectionStatusElement.className = 'status-indicator status-error';
        connectionTextElement.textContent = '已断开';
    });
    
    socket.on('config', function(config) {
        console.log('收到配置:', config);
        // 可以根据配置更新UI
    });
    
    // 处理分类结果
    socket.on('classification_result', function(dataStr) {
        try {
            const data = JSON.parse(dataStr);
            addClassificationMessage(data);
        } catch (error) {
            console.error('解析分类结果出错:', error);
        }
    });
    
    // 获取历史分类结果
    function fetchClassificationHistory() {
        fetch('/api/classification_history')
            .then(response => response.json())
            .then(data => {
                // 清空现有历史记录显示
                const classificationHistory = document.getElementById('classificationHistory');
                
                // 清空历史记录并更新
                if (classificationHistory) {
                    classificationHistory.innerHTML = '';
                    
                    if (data.history && data.history.length > 0) {
                        // 按时间戳排序（从新到旧）
                        const sortedHistory = [...data.history].sort((a, b) => b.timestamp - a.timestamp);
                        
                        // 更新显示区域
                        sortedHistory.forEach(item => {
                            // 更新大面板
                            addClassificationToHistory(item, classificationHistory);
                        });
                    } else {
                        // 如果没有历史记录，显示空消息
                        classificationHistory.innerHTML = '<div class="empty-message">暂无分类结果</div>';
                    }
                }
            })
            .catch(error => {
                console.error('获取分类历史记录失败:', error);
            });
    }
    
    // 添加新分类结果到UI
    function addClassificationMessage(data) {
        // 更新主分类历史面板
        const historyPanel = document.getElementById('classificationHistory');
        if (historyPanel) {
            if (historyPanel.querySelector('.empty-message')) {
                historyPanel.innerHTML = '';
            }
            addClassificationToHistory(data, historyPanel);
        }
    }
    
    // 不再需要添加分类到聊天框的函数，因为我们已经移除了聊天框
    
    // 添加分类到历史记录面板
    function addClassificationToHistory(item, container) {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'classification-item';
        
        const timeSpan = document.createElement('span');
        timeSpan.className = 'classification-time';
        timeSpan.textContent = item.formatted_time || new Date(item.timestamp).toLocaleString();
        
        const resultDiv = document.createElement('div');
        resultDiv.className = 'classification-result';
        resultDiv.textContent = item.classification;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'classification-message';
        messageDiv.textContent = item.message || `识别结果: ${item.classification}`;
        
        const confidenceSpan = document.createElement('span');
        confidenceSpan.className = 'classification-confidence';
        if (item.confidence !== undefined) {
            confidenceSpan.textContent = `置信度: ${(item.confidence * 100).toFixed(1)}%`;
        }
        
        itemDiv.appendChild(timeSpan);
        itemDiv.appendChild(resultDiv);
        itemDiv.appendChild(messageDiv);
        itemDiv.appendChild(confidenceSpan);
        
        // 添加到容器的顶部（最新的在顶部）
        if (container.firstChild) {
            container.insertBefore(itemDiv, container.firstChild);
        } else {
            container.appendChild(itemDiv);
        }
    }
    
    // 接收CSI数据
    socket.on('csi_data', function(dataStr) {
        if (!isRunning) return;
        
        try {
            const data = JSON.parse(dataStr);
            
            // 保存最新收到的数据，以便切换子载波或天线时使用
            lastReceivedData = data;
            
            // 调试输出
            console.log('收到CSI数据:', {
                timestamp: data.timestamp,
                type: data.type,
                has_amplitude_data: !!data.amplitude_data,
                has_phase_data: !!data.phase_data,
                has_subcarrier_data: !!data.subcarrier_data,
                has_classification: !!data.classification
            });
            
            // 检查是否是分类结果
            if (data.type === 'classification_result') {
                // 处理分类结果
                const classificationData = {
                    timestamp: data.timestamp,
                    formatted_time: new Date(data.timestamp).toLocaleString(),
                    classification: data.classification,
                    confidence: data.confidence,
                    message: `识别结果: ${data.classification}`
                };
                addClassificationMessage(classificationData);
                return;
            }
            
            // 更新显示的值
            currentSubcarrierElement.textContent = selectedSubcarrier;
            currentAntennasElement.textContent = `RX:${selectedRxAntenna} TX:${selectedTxAntenna}`;
            
            // 从完整数据中提取当前选择的天线和子载波的数据
            let currentAmplitude, currentPhase;

            // 如果有amplitude_data数据，从中提取当前选择的子载波和天线数据
            if (data.amplitude_data && Array.isArray(data.amplitude_data) && 
                data.amplitude_data.length > selectedRxAntenna &&
                data.amplitude_data[selectedRxAntenna].length > selectedTxAntenna && 
                data.amplitude_data[selectedRxAntenna][selectedTxAntenna].length > selectedSubcarrier) {

                currentAmplitude = data.amplitude_data[selectedRxAntenna][selectedTxAntenna][selectedSubcarrier];
                console.log(`从amplitude_data提取的幅值: ${currentAmplitude}`);
            }
            
            // 如果有csi_data，从中计算相位
            if (data.phase_data && Array.isArray(data.phase_data) && 
                data.phase_data.length > selectedRxAntenna && 
                data.phase_data[selectedRxAntenna].length > selectedTxAntenna &&
                data.phase_data[selectedRxAntenna][selectedTxAntenna].length > selectedSubcarrier) {
                
                currentPhase = data.phase_data[selectedRxAntenna][selectedTxAntenna][selectedSubcarrier];
                console.log(`从phase_data提取的相位: ${currentPhase}`);
            }
            
            // 记录幅值和相位值（不再更新DOM元素，因为它们已被删除）
            if (currentAmplitude !== undefined) {
                console.log(`当前幅值: ${currentAmplitude.toFixed(2)}`);
            } else {
                console.warn('无法获取当前幅值数据');
            }
            
            if (currentPhase !== undefined) {
                console.log(`当前相位: ${currentPhase.toFixed(2)}`);
            } else {
                console.warn('无法获取当前相位数据');
            }
            
            // 更新数据点计数
            dataPointCount++;
            
            // 更新图表数据
            const timestamp = new Date(data.timestamp || Date.now()).toLocaleTimeString();
            
            // 更新幅值图表
            if (currentAmplitude !== undefined) {
                amplitudeData.labels.shift();
                amplitudeData.labels.push(timestamp);
                amplitudeData.datasets[0].data.shift();
                amplitudeData.datasets[0].data.push(currentAmplitude);  // 使用当前选择的子载波和天线的幅值
                console.log("更新幅值图表:", currentAmplitude);
                amplitudeChart.update('none'); // 使用'none'模式加速更新
            } else {
                console.warn("幅值数据为undefined，跳过图表更新");
            }
            
            // 更新相位图表
            if (currentPhase !== undefined) {
                phaseData.labels.shift();
                phaseData.labels.push(timestamp);
                phaseData.datasets[0].data.shift();
                phaseData.datasets[0].data.push(currentPhase);  // 使用当前选择的子载波和天线的相位
                console.log("更新相位图表:", currentPhase);
                phaseChart.update('none'); // 使用'none'模式加速更新
            } else {
                console.warn("相位数据为undefined，跳过图表更新");
            }
            
            // SNR图表更新已移除
            
            // 更新频谱图
            // 优先使用subcarrier_data (已预处理的子载波数据)
            if (data.subcarrier_data) {
                const key = `rx${selectedRxAntenna}_tx${selectedTxAntenna}`;
                if (data.subcarrier_data[key]) {
                    spectrumData.datasets[0].data = data.subcarrier_data[key];
                    console.log(`使用subcarrier_data更新频谱图，选择的天线组合: ${key}`);
                    spectrumChart.update();
                } else {
                    console.warn(`subcarrier_data中不存在键 ${key}`);
                }
            }
            // 如果没有subcarrier_data，但有magnitude_data，直接使用magnitude_data
            else if (data.magnitude_data && Array.isArray(data.magnitude_data) && 
                    data.magnitude_data.length > selectedRxAntenna &&
                    data.magnitude_data[selectedRxAntenna].length > selectedTxAntenna) {
                
                // 直接从magnitude_data提取当前选择的RX/TX组合的所有子载波数据
                spectrumData.datasets[0].data = data.magnitude_data[selectedRxAntenna][selectedTxAntenna];
                console.log('使用magnitude_data更新频谱图');
                spectrumChart.update();
            } else {
                console.warn('无法更新频谱图，缺少子载波数据');
            }
            
            
            
        } catch (error) {
            console.error('处理CSI数据出错:', error);
        }
    });
    
    // 接收系统状态更新
    socket.on('status_update', function(status) {
        // 更新状态面板
        packetRateElement.textContent = status.packet_rate || 0;
        queueLengthElement.textContent = status.queue_length || 0;
        processedCountElement.textContent = status.processed_count || 0;
        processingRateElement.textContent = (status.processing_rate || 0).toFixed(2);
        
        // 更新运行时间
        const now = new Date();
        const diff = Math.floor((now - startTime) / 1000);
        const hours = Math.floor(diff / 3600);
        const minutes = Math.floor((diff % 3600) / 60);
        const seconds = diff % 60;
        uptimeElement.textContent = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        
        // 更新系统监控图表
        const timeStr = new Date().toLocaleTimeString();
        
        // 更新数据包速率图表
        packetRateData.labels.shift();
        packetRateData.labels.push(timeStr);
        packetRateData.datasets[0].data.shift();
        packetRateData.datasets[0].data.push(status.packet_rate || 0);
        packetRateChart.update();
        
        // 更新队列长度图表
        queueLengthData.labels.shift();
        queueLengthData.labels.push(timeStr);
        queueLengthData.datasets[0].data.shift();
        queueLengthData.datasets[0].data.push(status.queue_length || 0);
        queueLengthChart.update();
    });
    
    
    
    // 存储上一次收到的数据以便在切换子载波或天线时重新使用
    let lastReceivedData = null;
    
    // 更新当前视图，根据选择的子载波和天线重新计算显示数据
    function updateCurrentView() {
        if (!lastReceivedData) return;
        
        console.log(`更新视图: 子载波=${selectedSubcarrier}, RX=${selectedRxAntenna}, TX=${selectedTxAntenna}`);
        
        // 从上次接收的数据中提取当前选择的幅值
        let currentAmplitude, currentPhase;
        
        if (lastReceivedData.amplitude_data && 
            lastReceivedData.amplitude_data.length > selectedRxAntenna &&
            lastReceivedData.amplitude_data[selectedRxAntenna].length > selectedTxAntenna && 
            lastReceivedData.amplitude_data[selectedRxAntenna][selectedTxAntenna].length > selectedSubcarrier) {
            
            currentAmplitude = lastReceivedData.amplitude_data[selectedRxAntenna][selectedTxAntenna][selectedSubcarrier];
        }
        
        if (lastReceivedData.phase_data && 
            lastReceivedData.phase_data.length > selectedRxAntenna &&
            lastReceivedData.phase_data[selectedRxAntenna].length > selectedTxAntenna && 
            lastReceivedData.phase_data[selectedRxAntenna][selectedTxAntenna].length > selectedSubcarrier) {
            
            currentPhase = lastReceivedData.phase_data[selectedRxAntenna][selectedTxAntenna][selectedSubcarrier];
        }
        
        // 更新频谱图
        // if (lastReceivedData.subcarrier_data) {
        //     const key = `rx${selectedRxAntenna}_tx${selectedTxAntenna}`;
        //     if (lastReceivedData.subcarrier_data[key]) {
        //         spectrumData.datasets[0].data = lastReceivedData.subcarrier_data[key];
        //         spectrumChart.update();
        //     }
        // } else if (lastReceivedData.magnitude_data && 
        //            lastReceivedData.magnitude_data.length > selectedRxAntenna &&
        //            lastReceivedData.magnitude_data[selectedRxAntenna].length > selectedTxAntenna) {
            
        //     spectrumData.datasets[0].data = lastReceivedData.magnitude_data[selectedRxAntenna][selectedTxAntenna];
        //     spectrumChart.update();
        // }
    }
    
    // 事件处理程序
    subcarrierSelect.addEventListener('change', function() {
        selectedSubcarrier = parseInt(this.value);
        updateCurrentView();  // 立即更新视图
        sendControl();  // 发送控制命令到服务器
    });
    
    rxAntennaSelect.addEventListener('change', function() {
        selectedRxAntenna = parseInt(this.value);
        updateCurrentView();  // 立即更新视图
        sendControl();  // 发送控制命令到服务器
    });
    
    txAntennaSelect.addEventListener('change', function() {
        selectedTxAntenna = parseInt(this.value);
        updateCurrentView();  // 立即更新视图
        sendControl();  // 发送控制命令到服务器
    });
    
    displayModeSelect.addEventListener('change', function() {
        displayMode = this.value;
    });
    
    dataWindowSizeSelect.addEventListener('change', function() {
        MAX_DATA_POINTS = parseInt(this.value);
        resetCharts();
    });
    
    startButton.addEventListener('click', function() {
        isRunning = true;
    });
    
    pauseButton.addEventListener('click', function() {
        isRunning = false;
    });
    
    clearButton.addEventListener('click', function() {
        resetCharts();
        dataPointCount = 0;
        dataPointCountElement.textContent = dataPointCount;
    });
    
    // 发送控制命令到服务器
    function sendControl() {
        socket.emit('control', {
            subcarrier: selectedSubcarrier,
            rx_antenna: selectedRxAntenna,
            tx_antenna: selectedTxAntenna
        });
    }
    
    // 重置图表数据
    function resetCharts() {
        // 重置幅值图表
        amplitudeData.labels = Array(MAX_DATA_POINTS).fill('');
        amplitudeData.datasets[0].data = Array(MAX_DATA_POINTS).fill(null);
        amplitudeChart.update();
        
        // 重置相位图表
        phaseData.labels = Array(MAX_DATA_POINTS).fill('');
        phaseData.datasets[0].data = Array(MAX_DATA_POINTS).fill(null);
        phaseChart.update();
    }
    
    // 初始化
    sendControl();
    
    // 定期更新运行时间
    setInterval(function() {
        const now = new Date();
        const diff = Math.floor((now - startTime) / 1000);
        const hours = Math.floor(diff / 3600);
        const minutes = Math.floor((diff % 3600) / 60);
        const seconds = diff % 60;
        uptimeElement.textContent = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }, 1000);
});
