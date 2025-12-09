/**
 * Three.js 3D Model Viewer
 * 
 * Provides functionality to load and display 3D models (GLB, OBJ) in a web viewer.
 */

class ThreeViewer {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container not found: ${containerId}`);
        }
        
        this.options = {
            width: options.width || 800,
            height: options.height || 600,
            backgroundColor: options.backgroundColor || 0x1a1a1a,
            ...options,
        };
        
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.currentModel = null;
        this.animationId = null;
        
        this.init();
    }
    
    init() {
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(this.options.backgroundColor);
        
        // Create camera
        const aspect = this.options.width / this.options.height;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        this.camera.position.set(0, 0, 3);
        
        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.options.width, this.options.height);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);
        
        // Add lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 5, 5);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
        
        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight2.position.set(-5, -5, -5);
        this.scene.add(directionalLight2);
        
        // Add controls (if OrbitControls is available)
        // OrbitControls might be loaded as a separate script
        if (typeof THREE.OrbitControls !== 'undefined') {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.05;
            this.controls.minDistance = 0.5;
            this.controls.maxDistance = 10;
        } else if (typeof OrbitControls !== 'undefined') {
            // Try global OrbitControls
            this.controls = new OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.05;
            this.controls.minDistance = 0.5;
            this.controls.maxDistance = 10;
        }
        
        // Add grid helper
        const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
        this.scene.add(gridHelper);
        
        // Add axes helper
        const axesHelper = new THREE.AxesHelper(2);
        this.scene.add(axesHelper);
        
        // Start animation loop
        this.animate();
        
        // Handle window resize
        window.addEventListener('resize', () => this.handleResize());
    }
    
    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        
        if (this.controls) {
            this.controls.update();
        }
        
        this.renderer.render(this.scene, this.camera);
    }
    
    handleResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    
    async loadModel(url, fileType = 'glb') {
        try {
            // Remove existing model
            if (this.currentModel) {
                this.scene.remove(this.currentModel);
                // Dispose of geometry and materials
                this.disposeModel(this.currentModel);
                this.currentModel = null;
            }
            
            let model;
            
            if (fileType === 'glb' || fileType === 'gltf') {
                model = await this.loadGLB(url);
            } else if (fileType === 'obj') {
                model = await this.loadOBJ(url);
            } else {
                throw new Error(`Unsupported file type: ${fileType}`);
            }
            
            // Center and scale model
            this.fitModelToView(model);
            
            this.currentModel = model;
            this.scene.add(model);
            
            return model;
            
        } catch (error) {
            console.error('Failed to load 3D model:', error);
            throw error;
        }
    }
    
    async loadGLB(url) {
        return new Promise((resolve, reject) => {
            // Try THREE.GLTFLoader first, then global GLTFLoader
            const LoaderClass = THREE.GLTFLoader || (typeof GLTFLoader !== 'undefined' ? GLTFLoader : null);
            if (!LoaderClass) {
                reject(new Error('GLTFLoader not available. Please load Three.js GLTFLoader script.'));
                return;
            }
            
            const loader = new LoaderClass();
            loader.load(
                url,
                (gltf) => {
                    const model = gltf.scene || gltf;
                    resolve(model);
                },
                (progress) => {
                    // Progress callback
                    if (progress.lengthComputable) {
                        console.log('Loading progress:', (progress.loaded / progress.total * 100) + '%');
                    }
                },
                (error) => {
                    reject(error);
                }
            );
        });
    }
    
    async loadOBJ(url) {
        return new Promise((resolve, reject) => {
            // Try THREE.OBJLoader first, then global OBJLoader
            const LoaderClass = THREE.OBJLoader || (typeof OBJLoader !== 'undefined' ? OBJLoader : null);
            if (!LoaderClass) {
                reject(new Error('OBJLoader not available. Please load Three.js OBJLoader script.'));
                return;
            }
            
            const loader = new LoaderClass();
            loader.load(
                url,
                (object) => {
                    // OBJ loader returns a group, we'll use it directly
                    resolve(object);
                },
                (progress) => {
                    if (progress.lengthComputable) {
                        console.log('Loading progress:', (progress.loaded / progress.total * 100) + '%');
                    }
                },
                (error) => {
                    reject(error);
                }
            );
        });
    }
    
    fitModelToView(model) {
        // Calculate bounding box
        const box = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        
        // Center the model
        model.position.sub(center);
        
        // Scale to fit in view
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 2.0 / maxDim;
        model.scale.multiplyScalar(scale);
        
        // Adjust camera position
        const distance = maxDim * 1.5;
        this.camera.position.set(distance, distance, distance);
        this.camera.lookAt(0, 0, 0);
        
        if (this.controls) {
            this.controls.target.set(0, 0, 0);
            this.controls.update();
        }
    }
    
    disposeModel(model) {
        model.traverse((child) => {
            if (child.geometry) {
                child.geometry.dispose();
            }
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(material => material.dispose());
                } else {
                    child.material.dispose();
                }
            }
        });
    }
    
    setBackgroundColor(color) {
        this.scene.background = new THREE.Color(color);
    }
    
    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        if (this.currentModel) {
            this.disposeModel(this.currentModel);
            this.scene.remove(this.currentModel);
        }
        
        if (this.renderer) {
            this.renderer.dispose();
            if (this.container && this.renderer.domElement) {
                this.container.removeChild(this.renderer.domElement);
            }
        }
        
        if (this.controls) {
            this.controls.dispose();
        }
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ThreeViewer;
}

