<h1 align="center">Digit Classifier using NumPy</h1>

<p align="center">
  A simple two-layer neural network built from scratch in NumPy to classify handwritten digits from the MNIST dataset.
</p>

<hr>

<h2> Features</h2>

<ul>
  <li>Vectorized forward and backward propagation</li>
  <li>Manual ReLU and softmax activations</li>
  <li>Weight saving and loading with NumPy</li>
  <li>Dataset in CSV format, model fully framework-free</li>
</ul>

<h2> Project Structure</h2>

<pre>
digit-classifier-numpy/
├── train_model.py          # Training logic
├── test_model.py           # Model evaluation
├── download_mnist.py       # Downloads MNIST CSV
├── weights/                # Saved weights (W1, b1, W2, b2)
├── mnist_csv/              # CSV data files
└── README.md
</pre>

<h2> Dataset</h2>

<p>The project uses the MNIST dataset in CSV format.</p>

<h4>Manual Download from Kaggle</h4>

<ul>
  <li><a href="https://www.kaggle.com/datasets/oddrationale/mnist-in-csv">mnist_csv.csv</a></li>
</ul>

<p>Place them in a folder called <code>mnist_csv/</code>.</p>

<h2> Usage</h2>

<h4>Train the model</h4>

<pre><code>python train_model.py</code></pre>

<h4>Test the model</h4>

<pre><code>python test_model.py</code></pre>

<h2>🔧 Requirements</h2>

<table>
  <thead>
    <tr><th>Package</th><th>Purpose</th></tr>
  </thead>
  <tbody>
    <tr><td>NumPy</td><td>Matrix operations</td></tr>
    <tr><td>Pandas</td><td>CSV data handling</td></tr>
    <tr><td>Matplotlib</td><td>Optional: plot predictions</td></tr>
  </tbody>
</table>

<pre><code>pip install numpy pandas matplotlib</code></pre>

<h2> Notes</h2>

<ul>
  <li>Weights and datasets are ignored via <code>.gitignore</code></li>
  <li>All learning logic implemented from first principles</li>
</ul>

<h2> License</h2>

<p>This project is licensed under the MIT License.</p>