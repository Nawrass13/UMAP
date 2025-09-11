using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;

namespace UMAPuwotSharp
{
    /// <summary>
    /// Cross-platform C# wrapper for the UMAP dimensionality reduction library based on uwot
    /// </summary>
    public class UMapModel : IDisposable
    {
        #region Platform Detection and DLL Imports

        private static readonly bool IsWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        private static readonly bool IsLinux = RuntimeInformation.IsOSPlatform(OSPlatform.Linux);

        private const string WindowsDll = "uwot.dll";
        private const string LinuxDll = "libuwot.so";

        // Windows P/Invoke declarations
        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "uwot_create")]
        private static extern IntPtr WindowsCreate();

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "uwot_fit")]
        private static extern int WindowsFit(IntPtr model, float[] data, int nObs, int nDim, int nNeighbors, float minDist, int nEpochs, float[] embedding);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "uwot_transform")]
        private static extern int WindowsTransform(IntPtr model, float[] newData, int nNewObs, int nDim, float[] embedding);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "uwot_save_model")]
        private static extern int WindowsSaveModel(IntPtr model, [MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "uwot_load_model")]
        private static extern IntPtr WindowsLoadModel([MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "uwot_destroy")]
        private static extern void WindowsDestroy(IntPtr model);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "uwot_get_error_message")]
        private static extern IntPtr WindowsGetErrorMessage(int errorCode);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "uwot_get_model_info")]
        private static extern int WindowsGetModelInfo(IntPtr model, out int nVertices, out int nDim, out int embeddingDim, out int nNeighbors, out float minDist);

        [DllImport(WindowsDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "uwot_is_fitted")]
        private static extern int WindowsIsFitted(IntPtr model);

        // Linux P/Invoke declarations
        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "uwot_create")]
        private static extern IntPtr LinuxCreate();

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "uwot_fit")]
        private static extern int LinuxFit(IntPtr model, float[] data, int nObs, int nDim, int nNeighbors, float minDist, int nEpochs, float[] embedding);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "uwot_transform")]
        private static extern int LinuxTransform(IntPtr model, float[] newData, int nNewObs, int nDim, float[] embedding);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "uwot_save_model")]
        private static extern int LinuxSaveModel(IntPtr model, [MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "uwot_load_model")]
        private static extern IntPtr LinuxLoadModel([MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "uwot_destroy")]
        private static extern void LinuxDestroy(IntPtr model);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "uwot_get_error_message")]
        private static extern IntPtr LinuxGetErrorMessage(int errorCode);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "uwot_get_model_info")]
        private static extern int LinuxGetModelInfo(IntPtr model, out int nVertices, out int nDim, out int embeddingDim, out int nNeighbors, out float minDist);

        [DllImport(LinuxDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "uwot_is_fitted")]
        private static extern int LinuxIsFitted(IntPtr model);

        #endregion

        #region Error Codes

        private const int UWOT_SUCCESS = 0;
        private const int UWOT_ERROR_INVALID_PARAMS = -1;
        private const int UWOT_ERROR_MEMORY = -2;
        private const int UWOT_ERROR_NOT_IMPLEMENTED = -3;
        private const int UWOT_ERROR_FILE_IO = -4;
        private const int UWOT_ERROR_MODEL_NOT_FITTED = -5;
        private const int UWOT_ERROR_INVALID_MODEL_FILE = -6;

        #endregion

        #region Private Fields

        private IntPtr _nativeModel;
        private bool _disposed = false;

        #endregion

        #region Properties

        /// <summary>
        /// Gets whether the model has been fitted with training data
        /// </summary>
        public bool IsFitted => CallIsFitted(_nativeModel) != 0;

        /// <summary>
        /// Gets the model information if fitted
        /// </summary>
        public UMapModelInfo ModelInfo
        {
            get
            {
                if (!IsFitted)
                    throw new InvalidOperationException("Model must be fitted before accessing model info");

                var result = CallGetModelInfo(_nativeModel, out var nVertices, out var nDim, out var embeddingDim, out var nNeighbors, out var minDist);
                ThrowIfError(result);

                return new UMapModelInfo(nVertices, nDim, embeddingDim, nNeighbors, minDist);
            }
        }

        #endregion

        #region Constructors

        /// <summary>
        /// Creates a new UMAP model instance
        /// </summary>
        public UMapModel()
        {
            _nativeModel = CallCreate();
            if (_nativeModel == IntPtr.Zero)
                throw new OutOfMemoryException("Failed to create UMAP model");
        }

        /// <summary>
        /// Loads a UMAP model from a file
        /// </summary>
        /// <param name="filename">Path to the model file</param>
        public static UMapModel Load(string filename)
        {
            if (string.IsNullOrEmpty(filename))
                throw new ArgumentException("Filename cannot be null or empty", nameof(filename));

            if (!File.Exists(filename))
                throw new FileNotFoundException($"Model file not found: {filename}");

            var model = new UMapModel();
            model._nativeModel = CallLoadModel(filename);

            if (model._nativeModel == IntPtr.Zero)
            {
                model.Dispose();
                throw new InvalidDataException($"Failed to load model from file: {filename}");
            }

            return model;
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Fits the UMAP model to training data
        /// </summary>
        /// <param name="data">Training data as 2D array [samples, features]</param>
        /// <param name="nNeighbors">Number of nearest neighbors (default: 15)</param>
        /// <param name="minDist">Minimum distance between points in embedding (default: 0.1)</param>
        /// <param name="nEpochs">Number of optimization epochs (default: 200)</param>
        /// <returns>2D embedding coordinates [samples, 2]</returns>
        public float[,] Fit(float[,] data, int nNeighbors = 15, float minDist = 0.1f, int nEpochs = 200)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            var nSamples = data.GetLength(0);
            var nFeatures = data.GetLength(1);

            if (nSamples <= 0 || nFeatures <= 0)
                throw new ArgumentException("Data must have positive dimensions");

            if (nNeighbors <= 0 || nNeighbors >= nSamples)
                throw new ArgumentException("Number of neighbors must be positive and less than number of samples");

            if (minDist <= 0)
                throw new ArgumentException("Minimum distance must be positive");

            if (nEpochs <= 0)
                throw new ArgumentException("Number of epochs must be positive");

            // Flatten the input data
            var flatData = new float[nSamples * nFeatures];
            for (int i = 0; i < nSamples; i++)
            {
                for (int j = 0; j < nFeatures; j++)
                {
                    flatData[i * nFeatures + j] = data[i, j];
                }
            }

            // Prepare output array
            var embedding = new float[nSamples * 2];

            // Call native function
            var result = CallFit(_nativeModel, flatData, nSamples, nFeatures, nNeighbors, minDist, nEpochs, embedding);
            ThrowIfError(result);

            // Convert back to 2D array
            return ConvertTo2D(embedding, nSamples, 2);
        }

        /// <summary>
        /// Transforms new data using a fitted model
        /// </summary>
        /// <param name="newData">New data to transform [samples, features]</param>
        /// <returns>2D embedding coordinates [samples, 2]</returns>
        public float[,] Transform(float[,] newData)
        {
            if (newData == null)
                throw new ArgumentNullException(nameof(newData));

            if (!IsFitted)
                throw new InvalidOperationException("Model must be fitted before transforming new data");

            var nNewSamples = newData.GetLength(0);
            var nFeatures = newData.GetLength(1);

            if (nNewSamples <= 0 || nFeatures <= 0)
                throw new ArgumentException("New data must have positive dimensions");

            // Validate feature dimension matches training data
            var modelInfo = ModelInfo;
            if (nFeatures != modelInfo.InputDimension)
                throw new ArgumentException($"Feature dimension mismatch. Expected {modelInfo.InputDimension}, got {nFeatures}");

            // Flatten the input data
            var flatNewData = new float[nNewSamples * nFeatures];
            for (int i = 0; i < nNewSamples; i++)
            {
                for (int j = 0; j < nFeatures; j++)
                {
                    flatNewData[i * nFeatures + j] = newData[i, j];
                }
            }

            // Prepare output array
            var embedding = new float[nNewSamples * 2];

            // Call native function
            var result = CallTransform(_nativeModel, flatNewData, nNewSamples, nFeatures, embedding);
            ThrowIfError(result);

            // Convert back to 2D array
            return ConvertTo2D(embedding, nNewSamples, 2);
        }

        /// <summary>
        /// Saves the fitted model to a file
        /// </summary>
        /// <param name="filename">Path where to save the model</param>
        public void Save(string filename)
        {
            if (string.IsNullOrEmpty(filename))
                throw new ArgumentException("Filename cannot be null or empty", nameof(filename));

            if (!IsFitted)
                throw new InvalidOperationException("Model must be fitted before saving");

            // Ensure directory exists
            var directory = Path.GetDirectoryName(filename);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                Directory.CreateDirectory(directory);

            var result = CallSaveModel(_nativeModel, filename);
            ThrowIfError(result);
        }

        #endregion

        #region Private Platform-Specific Wrappers

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static IntPtr CallCreate()
        {
            return IsWindows ? WindowsCreate() : LinuxCreate();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallFit(IntPtr model, float[] data, int nObs, int nDim, int nNeighbors, float minDist, int nEpochs, float[] embedding)
        {
            return IsWindows ? WindowsFit(model, data, nObs, nDim, nNeighbors, minDist, nEpochs, embedding)
                             : LinuxFit(model, data, nObs, nDim, nNeighbors, minDist, nEpochs, embedding);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallTransform(IntPtr model, float[] newData, int nNewObs, int nDim, float[] embedding)
        {
            return IsWindows ? WindowsTransform(model, newData, nNewObs, nDim, embedding)
                             : LinuxTransform(model, newData, nNewObs, nDim, embedding);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallSaveModel(IntPtr model, string filename)
        {
            return IsWindows ? WindowsSaveModel(model, filename) : LinuxSaveModel(model, filename);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static IntPtr CallLoadModel(string filename)
        {
            return IsWindows ? WindowsLoadModel(filename) : LinuxLoadModel(filename);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CallDestroy(IntPtr model)
        {
            if (IsWindows) WindowsDestroy(model);
            else LinuxDestroy(model);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static string CallGetErrorMessage(int errorCode)
        {
            var ptr = IsWindows ? WindowsGetErrorMessage(errorCode) : LinuxGetErrorMessage(errorCode);
            return Marshal.PtrToStringAnsi(ptr) ?? "Unknown error";
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallGetModelInfo(IntPtr model, out int nVertices, out int nDim, out int embeddingDim, out int nNeighbors, out float minDist)
        {
            return IsWindows ? WindowsGetModelInfo(model, out nVertices, out nDim, out embeddingDim, out nNeighbors, out minDist)
                             : LinuxGetModelInfo(model, out nVertices, out nDim, out embeddingDim, out nNeighbors, out minDist);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CallIsFitted(IntPtr model)
        {
            return IsWindows ? WindowsIsFitted(model) : LinuxIsFitted(model);
        }

        #endregion

        #region Utility Methods

        private static void ThrowIfError(int errorCode)
        {
            if (errorCode == UWOT_SUCCESS) return;

            var message = CallGetErrorMessage(errorCode);

            throw errorCode switch
            {
                UWOT_ERROR_INVALID_PARAMS => new ArgumentException(message),
                UWOT_ERROR_MEMORY => new OutOfMemoryException(message),
                UWOT_ERROR_NOT_IMPLEMENTED => new NotImplementedException(message),
                UWOT_ERROR_FILE_IO => new IOException(message),
                UWOT_ERROR_MODEL_NOT_FITTED => new InvalidOperationException(message),
                UWOT_ERROR_INVALID_MODEL_FILE => new InvalidDataException(message),
                _ => new Exception($"UMAP Error ({errorCode}): {message}")
            };
        }

        private static float[,] ConvertTo2D(float[] flatArray, int rows, int cols)
        {
            var result = new float[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = flatArray[i * cols + j];
                }
            }
            return result;
        }

        #endregion

        #region IDisposable Implementation

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed && _nativeModel != IntPtr.Zero)
            {
                CallDestroy(_nativeModel);
                _nativeModel = IntPtr.Zero;
                _disposed = true;
            }
        }

        ~UMapModel()
        {
            Dispose(false);
        }

        #endregion
    }

    /// <summary>
    /// Information about a fitted UMAP model
    /// </summary>
    public readonly struct UMapModelInfo
    {
        public int TrainingSamples { get; }
        public int InputDimension { get; }
        public int OutputDimension { get; }
        public int Neighbors { get; }
        public float MinimumDistance { get; }

        internal UMapModelInfo(int trainingSamples, int inputDimension, int outputDimension, int neighbors, float minimumDistance)
        {
            TrainingSamples = trainingSamples;
            InputDimension = inputDimension;
            OutputDimension = outputDimension;
            Neighbors = neighbors;
            MinimumDistance = minimumDistance;
        }

        public override string ToString()
        {
            return $"UMAP Model: {TrainingSamples} samples, {InputDimension}D ? {OutputDimension}D, k={Neighbors}, min_dist={MinimumDistance:F3}";
        }
    }
}