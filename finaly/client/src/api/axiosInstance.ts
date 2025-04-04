import axios from "axios"

export const BASE_URL = ''

const axiosInstance = axios.create({
    baseURL: BASE_URL,
    withCredentials: true,
})

export default axiosInstance
