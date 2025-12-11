/**
 * Workflow State Management
 * 
 * Centralized state management for the workflow application.
 * All modules should access state through this module.
 */

const WorkflowState = {
  // Workflow state
  currentFrameDir: null,
  currentSessionId: null,
  currentTrainingJobId: null,
  currentUploadId: null,
  currentVideoName: null,

  // Video metadata
  holdColor: '',
  routeDifficulty: '',
  firstFrameImageUrl: null,
  frameAspectRatio: null, // 'vertical' or 'horizontal'

  // 3D viewer (currently disabled)
  threeViewer: null,
  current3DModelId: null,

  // Frame selection state
  frameSelectionState: {
    uploadId: null,
    videoName: null,
    frames: [],
    currentIndex: 0,
    selectedFrames: new Set(),
    viewMode: 'all', // 'all' or 'selected'
  },

  // Hold labeling state
  holdLabelingSegments: [],
  holdLabelingCanvas: null,
  holdLabelingCtx: null,
  holdLabelingImage: null,
  selectedSegmentId: null,
  selectedSegmentIds: new Set(),

  // Getters
  getCurrentFrameDir() {
    return this.currentFrameDir;
  },

  getCurrentSessionId() {
    return this.currentSessionId;
  },

  getCurrentTrainingJobId() {
    return this.currentTrainingJobId;
  },

  getCurrentUploadId() {
    return this.currentUploadId;
  },

  getCurrentVideoName() {
    return this.currentVideoName;
  },

  getFrameSelectionState() {
    return this.frameSelectionState;
  },

  getFrameAspectRatio() {
    return this.frameAspectRatio;
  },

  getHoldLabelingSegments() {
    return this.holdLabelingSegments;
  },

  // Setters
  setCurrentFrameDir(value) {
    this.currentFrameDir = value;
  },

  setCurrentSessionId(value) {
    this.currentSessionId = value;
  },

  setCurrentTrainingJobId(value) {
    this.currentTrainingJobId = value;
  },

  setCurrentUploadId(value) {
    this.currentUploadId = value;
  },

  setCurrentVideoName(value) {
    this.currentVideoName = value;
  },

  setHoldColor(value) {
    this.holdColor = value;
  },

  setRouteDifficulty(value) {
    this.routeDifficulty = value;
  },

  setFirstFrameImageUrl(value) {
    this.firstFrameImageUrl = value;
  },

  setFrameAspectRatio(value) {
    this.frameAspectRatio = value;
  },

  getFrameAspectRatio() {
    return this.frameAspectRatio;
  },

  // Reset methods
  resetFrameSelection() {
    this.frameSelectionState = {
      uploadId: null,
      videoName: null,
      frames: [],
      currentIndex: 0,
      selectedFrames: new Set(),
      viewMode: 'all',
    };
  },

  resetHoldLabeling() {
    this.holdLabelingSegments = [];
    this.holdLabelingCanvas = null;
    this.holdLabelingCtx = null;
    this.holdLabelingImage = null;
    this.selectedSegmentId = null;
    this.selectedSegmentIds = new Set();
  },

  reset() {
    this.currentFrameDir = null;
    this.currentSessionId = null;
    this.currentTrainingJobId = null;
    this.currentUploadId = null;
    this.currentVideoName = null;
    this.holdColor = '';
    this.routeDifficulty = '';
    this.firstFrameImageUrl = null;
    this.frameAspectRatio = null;
    this.resetFrameSelection();
    this.resetHoldLabeling();
  }
};
