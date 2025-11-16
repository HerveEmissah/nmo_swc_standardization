# pull official base image
FROM node:13.12.0-alpine

# set working directory
WORKDIR /app

# add `/app/node_modules/.bin` to $PATH
ENV PATH /app/node_modules/.bin:$PATH

# install app dependencies
COPY package.json ./
#COPY package-lock.json ./
RUN npm install --silent
RUN npm install react-scripts@3.4.1 -g --silent
RUN npm install @material-ui/core --silent
RUN npm install @material-ui/lab --silent
RUN npm install @material-ui/styles --silent
RUN npm install @material-ui/icons --silent
RUN npm install autosuggest-highlight --silent
RUN npm install socket.io-client --silent

# add app
COPY . ./

# Set environment variables
ENV CHOKIDAR_USEPOLLING=true
ENV DANGEROUSLY_DISABLE_HOST_CHECK=true
ENV HOST=0.0.0.0
ENV PORT 80

# start the app 
CMD ["npm", "start"]
#CMD ["npm", "start", "--", "--port", "8899"]

