import * as Prometheus from 'prom-client';

// Initialize default metrics collection
Prometheus.collectDefaultMetrics();

// Custom metrics for frontend
export const frontendMetrics = {
  requestCounter: new Prometheus.Counter({
    name: 'frontend_requests_total',
    help: 'Total frontend requests',
    labelNames: ['method', 'endpoint']
  }),

  processingTime: new Prometheus.Histogram({
    name: 'frontend_processing_time_seconds',
    help: 'Frontend processing time',
    labelNames: ['method', 'endpoint']
  }),

  trackRequest: (method, endpoint, callback) => {
    const startTime = Date.now();
    frontendMetrics.requestCounter.labels(method, endpoint).inc();

    try {
      const result = callback();
      
      frontendMetrics.processingTime.labels(method, endpoint)
        .observe((Date.now() - startTime) / 1000);
      
      return result;
    } catch (error) {
      console.error('Request tracking error:', error);
      throw error;
    }
  },

  // Optional: Expose metrics for scraping
  exportMetrics: () => Prometheus.register.metrics()
}; 