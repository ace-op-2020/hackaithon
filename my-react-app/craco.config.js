const webpack = require('webpack');

module.exports = {
  webpack: {
    configure: (webpackConfig) => {
      // Add fallbacks for Node.js modules
      webpackConfig.resolve.fallback = {
        ...webpackConfig.resolve.fallback,
        "buffer": require.resolve("buffer/"),
        "crypto": require.resolve("crypto-browserify"),
        "stream": require.resolve("stream-browserify"),
        "https": require.resolve("https-browserify"),
        "http": require.resolve("stream-http"),
        "assert": require.resolve("assert/"),
        "path": require.resolve("path-browserify"),
        "os": require.resolve("os-browserify/browser"),
        "querystring": require.resolve("querystring-es3"),
        "url": require.resolve("url/"),
        "util": require.resolve("util/"),
        "fs": false,
        "net": false,
        "tls": false,
        "child_process": false
      };

      // Add plugins
      webpackConfig.plugins = [
        ...webpackConfig.plugins,
        new webpack.ProvidePlugin({
          Buffer: ['buffer', 'Buffer'],
          process: 'process/browser',
        }),
      ];

      // Handle node: scheme imports
      webpackConfig.resolve.alias = {
        ...webpackConfig.resolve.alias,
        "node:events": "events",
        "node:process": "process/browser",
        "node:util": "util",
        "node:buffer": "buffer",
        "node:stream": "stream-browserify"
      };

      return webpackConfig;
    },
  },
};
