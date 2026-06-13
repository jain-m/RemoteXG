export async function handler(request, context) {
  if (request.method !== 'POST') {
    return new Response('Please send a POST request with your data matrix', { status: 400 });
  }

  try {
    const body = await request.json();
    const { X, y, max_depth = 3, learning_rate = 0.3, n_estimators = 100 } = body;

    if (!X || !y || X.length !== y.length) {
      return new Response(JSON.stringify({ error: "Invalid input matrix dimensions" }), { status: 400 });
    }

    // Initialize predictions with the average of y
    const base_pred = y.reduce((a, b) => a + b, 0) / y.length;
    let preds = new Array(y.length).fill(base_pred);

    // Simple Decision Stump / Tree Builder for MSE regression
    function buildTree(X, residuals, depth) {
      if (depth >= max_depth || X.length <= 1) {
        return { value: residuals.reduce((a, b) => a + b, 0) / residuals.length };
      }

      let bestSplit = { feature: -1, threshold: 0, score: Infinity, leftIdx: [], rightIdx: [] };
      const numFeatures = X[0].length;

      for (let f = 0; f < numFeatures; f++) {
        const values = X.map(row => row[f]);
        for (const threshold of values) {
          let leftIdx = [], rightIdx = [];
          for (let i = 0; i < X.length; i++) {
            if (X[i][f] <= threshold) leftIdx.push(i);
            else rightIdx.push(i);
          }

          if (leftIdx.length === 0 || rightIdx.length === 0) continue;

          const leftRes = leftIdx.map(i => residuals[i]);
          const rightRes = rightIdx.map(i => residuals[i]);

          const leftMean = leftRes.reduce((a,b)=>a+b,0)/leftRes.length;
          const rightMean = rightRes.reduce((a,b)=>a+b,0)/rightRes.length;

          const score = leftRes.reduce((a,b)=>a+Math.pow(b-leftMean,2),0) + rightRes.reduce((a,b)=>a+Math.pow(b-rightMean,2),0);

          if (score < bestSplit.score) {
            bestSplit = { feature: f, threshold, score, leftIdx, rightIdx };
          }
        }
      }

      if (bestSplit.feature === -1) {
        return { value: residuals.reduce((a, b) => a + b, 0) / residuals.length };
      }

      return {
        feature: bestSplit.feature,
        threshold: bestSplit.threshold,
        left: buildTree(bestSplit.leftIdx.map(i => X[i]), bestSplit.leftIdx.map(i => residuals[i]), depth + 1),
        right: buildTree(bestSplit.rightIdx.map(i => X[i]), bestSplit.rightIdx.map(i => residuals[i]), depth + 1)
      };
    }

    function predictTree(tree, row) {
      if (tree.value !== undefined) return tree.value;
      if (row[tree.feature] <= tree.threshold) return predictTree(tree.left, row);
      return predictTree(tree.right, row);
    }

    const trees = [];
    // Boosting loop
    for (let t = 0; t < n_estimators; t++) {
      const residuals = y.map((val, i) => val - preds[i]);
      const tree = buildTree(X, residuals, 0);
      
      for (let i = 0; i < X.length; i++) {
        preds[i] += learning_rate * predictTree(tree, X[i]);
      }
      trees.push(tree);
    }

    // Calculate final Mean Squared Error
    const mse = y.reduce((sum, val, i) => sum + Math.pow(val - preds[i], 2), 0) / y.length;

    // Package the trained model structure into a Base64 string payload
    const modelPayload = JSON.stringify({ base_value: base_pred, trees: trees });
    const b64Model = Buffer.from(modelPayload).toString('base64');

    return new Response(JSON.stringify({
      status: "success",
      metrics: { final_mse: mse, iterations: n_estimators },
      model_b64: b64Model
    }), {
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}