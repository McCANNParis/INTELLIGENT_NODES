/**
 * Automated Bayesian Optimization Runner for ComfyUI
 * 
 * This script automates the execution of 20 iterations for Bayesian optimization
 * with automatic image comparison and parameter improvement.
 * 
 * Usage:
 * 1. Load the bayesian-fully-automated.json workflow in ComfyUI
 * 2. Set the PrimitiveNode (is_first_run) to true
 * 3. Upload your target image in the LoadImage node
 * 4. Run the workflow once manually (this is iteration 1)
 * 5. Set the PrimitiveNode (is_first_run) to false
 * 6. Open browser console (F12) and paste this script
 * 7. Call: runBayesianOptimization(19) to run iterations 2-20
 */

function runBayesianOptimization(iterations = 19, delaySeconds = 10) {
    console.log(`Starting Bayesian optimization for ${iterations} iterations...`);
    console.log(`Delay between iterations: ${delaySeconds} seconds`);
    
    let completedCount = 0;
    let failedCount = 0;
    
    // Function to queue a single iteration
    function queueIteration(iterationNum) {
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                try {
                    // Find and click the queue button
                    const queueButton = document.querySelector('#queue-button');
                    if (!queueButton) {
                        console.error('Queue button not found!');
                        reject('Queue button not found');
                        return;
                    }
                    
                    queueButton.click();
                    completedCount++;
                    
                    // Calculate actual iteration number (add 1 because first was manual)
                    const actualIteration = iterationNum + 2;
                    console.log(`✓ Queued iteration ${actualIteration}/20 (${completedCount}/${iterations} automated)`);
                    
                    // Show progress bar
                    const progress = Math.round((completedCount / iterations) * 100);
                    const progressBar = '█'.repeat(Math.floor(progress / 5)) + '░'.repeat(20 - Math.floor(progress / 5));
                    console.log(`Progress: [${progressBar}] ${progress}%`);
                    
                    resolve();
                } catch (error) {
                    failedCount++;
                    console.error(`✗ Failed to queue iteration ${iterationNum + 2}: ${error}`);
                    reject(error);
                }
            }, iterationNum * delaySeconds * 1000);
        });
    }
    
    // Queue all iterations
    const promises = [];
    for (let i = 0; i < iterations; i++) {
        promises.push(queueIteration(i));
    }
    
    // Wait for all iterations to complete
    Promise.allSettled(promises).then(results => {
        console.log('\\n=== Bayesian Optimization Complete ===');
        console.log(`✓ Successfully queued: ${completedCount} iterations`);
        if (failedCount > 0) {
            console.log(`✗ Failed: ${failedCount} iterations`);
        }
        console.log(`Total iterations run: ${completedCount + 1}/20 (including initial manual run)`);
        console.log('\\nResults saved to: /workspace/ComfyUI/output/bayesian_iterations/');
        console.log('Check the optimization dashboard for parameter improvements!');
    });
    
    // Return a control object
    return {
        stop: function() {
            console.log('Stopping optimization...');
            // Clear all pending timeouts
            for (let i = 0; i < iterations; i++) {
                clearTimeout(i);
            }
        },
        status: function() {
            console.log(`Status: ${completedCount}/${iterations} completed, ${failedCount} failed`);
        }
    };
}

// Quick start commands
console.log('=== Bayesian Optimization Automation Ready ===');
console.log('Commands:');
console.log('  runBayesianOptimization(19)     - Run 19 iterations (2-20) with 10s delay');
console.log('  runBayesianOptimization(19, 15) - Run 19 iterations with 15s delay');
console.log('  runBayesianOptimization(5, 5)   - Run 5 iterations with 5s delay (for testing)');
console.log('');
console.log('Make sure:');
console.log('1. You have already run iteration 1 manually');
console.log('2. is_first_run is set to false');
console.log('3. Your target image is loaded');