// 主應用類
class GLSLRayTracer {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.gl = null;
        this.program = null;
        this.uniformLocations = {};
        this.attributeLocations = {};
        this.buffers = {};
        
        this.camera = new CameraController();
        this.performanceMonitor = new PerformanceMonitor();
        
        // 渲染參數
        this.params = {
            cameraX: 0,
            cameraY: 2.5,
            cameraZ: 3.5,
            sphereY: 0.5,
            noiseFreq: 4.0,
            time: 0
        };
        
        this.startTime = Date.now();
        this.isRunning = false;
        
        this.init();
    }
    
    init() {
        try {
            this.initWebGL();
            this.initShaders();
            this.initBuffers();
            this.initUniforms();
            this.setupEventListeners();
            this.resize();
            this.start();
            
            console.log('GLSL Ray Tracer 初始化成功');
        } catch (error) {
            console.error('初始化失敗:', error);
            this.showError(error.message);
        }
    }
    
    initWebGL() {
        this.gl = this.canvas.getContext('webgl') || this.canvas.getContext('experimental-webgl');
        
        if (!this.gl) {
            throw new Error('WebGL 不支援。請使用支援WebGL的瀏覽器。');
        }
        
        // 檢查所需擴展
        const ext = this.gl.getExtension('OES_standard_derivatives');
        if (!ext) {
            console.warn('OES_standard_derivatives 擴展不可用');
        }
        
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.gl.clearColor(0.0, 0.0, 0.0, 1.0);
    }
    
    initShaders() {
        this.program = ShaderUtils.createProgram(this.gl, vertexShaderSource, fragmentShaderSource);
        this.gl.useProgram(this.program);
    }
    
    initBuffers() {
        // 建立全螢幕四邊形
        const vertices = [
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0
        ];
        
        this.buffers.position = ShaderUtils.createBuffer(this.gl, vertices);
    }
    
    initUniforms() {
        // 取得uniform位置
        this.uniformLocations = {
            time: ShaderUtils.getUniformLocation(this.gl, this.program, 'u_time'),
            resolution: ShaderUtils.getUniformLocation(this.gl, this.program, 'u_resolution'),
            cameraPos: ShaderUtils.getUniformLocation(this.gl, this.program, 'u_cameraPos'),
            sphereY: ShaderUtils.getUniformLocation(this.gl, this.program, 'u_sphereY'),
            noiseFreq: ShaderUtils.getUniformLocation(this.gl, this.program, 'u_noiseFreq')
        };
        
        // 取得attribute位置
        this.attributeLocations = {
            position: ShaderUtils.getAttribLocation(this.gl, this.program, 'a_position')
        };
        
        // 設定頂點屬性
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.position);
        ShaderUtils.setupVertexAttribute(this.gl, this.attributeLocations.position, 2);
    }
    
    setupEventListeners() {
        // 視窗大小變化
        window.addEventListener('resize', () => this.resize());
        
        // 控制滑桿
        const controls = {
            cameraX: document.getElementById('cameraX'),
            cameraY: document.getElementById('cameraY'),
            cameraZ: document.getElementById('cameraZ'),
            sphereY: document.getElementById('sphereY'),
            noiseFreq: document.getElementById('noiseFreq')
        };
        
        Object.keys(controls).forEach(key => {
            const element = controls[key];
            if (element) {
                element.addEventListener('input', (e) => {
                    this.params[key] = parseFloat(e.target.value);
                });
                
                // 設定初始值
                this.params[key] = parseFloat(element.value);
            }
        });
        
        // 重置按鈕
        const resetBtn = document.getElementById('resetBtn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.reset());
        }
        
        // 鍵盤控制
        document.addEventListener('keydown', (e) => this.handleKeyDown(e));
        
        // 滑鼠控制
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        this.canvas.addEventListener('wheel', (e) => this.handleWheel(e));
        
        this.mouseState = {
            isDown: false,
            lastX: 0,
            lastY: 0
        };
    }
    
    handleKeyDown(e) {
        const speed = 0.1;
        
        switch(e.key.toLowerCase()) {
            case 'w':
                this.params.cameraZ -= speed;
                break;
            case 's':
                this.params.cameraZ += speed;
                break;
            case 'a':
                this.params.cameraX -= speed;
                break;
            case 'd':
                this.params.cameraX += speed;
                break;
            case 'q':
                this.params.cameraY += speed;
                break;
            case 'e':
                this.params.cameraY -= speed;
                break;
            case 'r':
                this.reset();
                break;
        }
        
        this.updateControls();
    }
    
    handleMouseDown(e) {
        this.mouseState.isDown = true;
        this.mouseState.lastX = e.clientX;
        this.mouseState.lastY = e.clientY;
    }
    
    handleMouseMove(e) {
        if (!this.mouseState.isDown) return;
        
        const deltaX = e.clientX - this.mouseState.lastX;
        const deltaY = e.clientY - this.mouseState.lastY;
        
        const sensitivity = 0.01;
        this.params.cameraX += deltaX * sensitivity;
        this.params.cameraY -= deltaY * sensitivity;
        
        this.mouseState.lastX = e.clientX;
        this.mouseState.lastY = e.clientY;
        
        this.updateControls();
    }
    
    handleMouseUp(e) {
        this.mouseState.isDown = false;
    }
    
    handleWheel(e) {
        e.preventDefault();
        const delta = e.deltaY > 0 ? 0.1 : -0.1;
        this.params.cameraZ = Math.max(0.5, Math.min(10, this.params.cameraZ + delta));
        this.updateControls();
    }
    
    updateControls() {
        // 更新滑桿值
        const controls = ['cameraX', 'cameraY', 'cameraZ', 'sphereY', 'noiseFreq'];
        controls.forEach(name => {
            const element = document.getElementById(name);
            if (element) {
                element.value = this.params[name];
            }
        });
    }
    
    reset() {
        this.params = {
            cameraX: 0,
            cameraY: 2.5,
            cameraZ: 3.5,
            sphereY: 0.5,
            noiseFreq: 4.0,
            time: 0
        };
        
        this.updateControls();
    }
    
    resize() {
        const displayWidth = this.canvas.clientWidth;
        const displayHeight = this.canvas.clientHeight;
        
        if (this.canvas.width !== displayWidth || this.canvas.height !== displayHeight) {
            this.canvas.width = displayWidth;
            this.canvas.height = displayHeight;
            
            if (this.gl) {
                this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
            }
        }
    }
    
    updateUniforms() {
        const gl = this.gl;
        
        // 更新時間
        this.params.time = (Date.now() - this.startTime) / 1000.0;
        
        // 設定uniforms
        if (this.uniformLocations.time !== null) {
            gl.uniform1f(this.uniformLocations.time, this.params.time);
        }
        
        if (this.uniformLocations.resolution !== null) {
            gl.uniform2f(this.uniformLocations.resolution, this.canvas.width, this.canvas.height);
        }
        
        if (this.uniformLocations.cameraPos !== null) {
            gl.uniform3f(this.uniformLocations.cameraPos, 
                this.params.cameraX, this.params.cameraY, this.params.cameraZ);
        }
        
        if (this.uniformLocations.sphereY !== null) {
            gl.uniform1f(this.uniformLocations.sphereY, this.params.sphereY);
        }
        
        if (this.uniformLocations.noiseFreq !== null) {
            gl.uniform1f(this.uniformLocations.noiseFreq, this.params.noiseFreq);
        }
    }
    
    render() {
        if (!this.isRunning) return;
        
        this.performanceMonitor.update();
        
        const gl = this.gl;
        
        // 清除螢幕
        gl.clear(gl.COLOR_BUFFER_BIT);
        
        // 使用程式
        gl.useProgram(this.program);
        
        // 更新uniforms
        this.updateUniforms();
        
        // 綁定頂點緩衝區
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.position);
        gl.enableVertexAttribArray(this.attributeLocations.position);
        gl.vertexAttribPointer(this.attributeLocations.position, 2, gl.FLOAT, false, 0, 0);
        
        // 渲染四邊形
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        
        // 繼續下一幀
        requestAnimationFrame(() => this.render());
    }
    
    start() {
        if (this.isRunning) return;
        
        this.isRunning = true;
        this.startTime = Date.now();
        this.render();
        
        console.log('Ray Tracer 開始執行');
    }
    
    stop() {
        this.isRunning = false;
        console.log('Ray Tracer 停止執行');
    }
    
    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #ff4444;
            color: white;
            padding: 20px;
            border-radius: 8px;
            font-family: monospace;
            z-index: 1000;
        `;
        errorDiv.innerHTML = `
            <h3>錯誤</h3>
            <p>${message}</p>
            <button onclick="this.parentElement.remove()">關閉</button>
        `;
        document.body.appendChild(errorDiv);
    }
}

// 應用初始化
document.addEventListener('DOMContentLoaded', () => {
    try {
        const app = new GLSLRayTracer();
        
        // 新增效能顯示
        const perfDiv = document.createElement('div');
        perfDiv.style.cssText = `
            position: fixed;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.8);
            color: #4CAF50;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            z-index: 1000;
        `;
        document.body.appendChild(perfDiv);
        
        setInterval(() => {
            perfDiv.innerHTML = `
                FPS: ${app.performanceMonitor.getFPS()}<br>
                幀時間: ${app.performanceMonitor.getFrameTime()}ms
            `;
        }, 1000);
        
        // 全域參考，便於除錯
        window.rayTracer = app;
        
    } catch (error) {
        console.error('應用啟動失敗:', error);
    }
}); 