#include <iostream>
#include <libusb.h>
#include <stdio.h>
#include <string>
#include <map>
#include <set>

using namespace std;

// Карта для преобразования Vendor ID в название производителя
map<uint16_t, string> vendor_names = {
    {0x0408, "Quanta Computer Inc."},
    {0x1358, "Realtek Semiconductor Corp."},
    {0x27c6, "Shenzhen Goodix Technology Co., Ltd."},
    {0x1022, "Advanced Micro Devices, Inc."},
    {0x8087, "Intel Corporation"},
    {0x413c, "Dell Inc."},
    {0x046d, "Logitech Inc."},
    {0x045e, "Microsoft Corporation"},
    {0x04f2, "Chicony Electronics Co., Ltd."},
    {0x0bda, "Realtek Semiconductor Corp."},
    {0x1d6b, "Linux Foundation"}
};

// Карта для преобразования Product ID в название устройства
map<uint16_t, map<uint16_t, string>> product_names = {
    {0x1358, {{0xc123, "RTX Bluetooth Radio"}}},
    {0x27c6, {{0x5125, "Fingerprint Reader"}}},
    {0x1022, {{0x1639, "USB 3.0 Hub"}}},
    {0x0408, {{0x1040, "Integrated Webcam"}}}
};

// Функция для получения описания класса устройства
string get_device_class_name(uint8_t device_class) {
    switch(device_class) {
        case 0x00: return "Interface-specific";
        case 0x01: return "Audio";
        case 0x02: return "Network";
        case 0x03: return "HID";
        case 0x07: return "Printer";
        case 0x08: return "Mass Storage";
        case 0x09: return "USB Hub";
        case 0x0A: return "CDC Data";
        case 0x0B: return "Smart Card";
        case 0x0D: return "Content Security";
        case 0x0E: return "Video";
        case 0x0F: return "Personal Healthcare";
        case 0x10: return "Audio/Video";
        case 0xDC: return "Diagnostic";
        case 0xE0: return "Wireless Controller";
        case 0xEF: return "Miscellaneous";
        case 0xFE: return "Application Specific";
        case 0xFF: return "Vendor Specific";
        default: return "Unknown";
    }
}

// Функция для получения названия производителя
string get_vendor_name(uint16_t vendor_id) {
    auto it = vendor_names.find(vendor_id);
    return (it != vendor_names.end()) ? it->second : "Unknown Vendor";
}

// Функция для получения названия продукта
string get_product_name(uint16_t vendor_id, uint16_t product_id) {
    auto vendor_it = product_names.find(vendor_id);
    if (vendor_it != product_names.end()) {
        auto product_it = vendor_it->second.find(product_id);
        if (product_it != vendor_it->second.end()) {
            return product_it->second;
        }
    }
    return "Unknown Device";
}

// Функция для печати серийного номера
void print_serial_number(libusb_device *dev) {
    libusb_device_descriptor desc;
    if (libusb_get_device_descriptor(dev, &desc) < 0) return;

    if (desc.iSerialNumber == 0) {
        cout << "Not available" << endl;
        return;
    }

    libusb_device_handle *handle;
    int r = libusb_open(dev, &handle);

    if (r != 0) {
        cout << "Access denied" << endl;
        return;
    }

    unsigned char serial[256];
    int bytes = libusb_get_string_descriptor_ascii(handle, desc.iSerialNumber, serial, sizeof(serial));
    libusb_close(handle);

    if (bytes > 0) {
        cout << serial << endl;
    } else {
        cout << "Read error" << endl;
    }
}

// Функция для печати информации об устройстве
void print_device_info(libusb_device *dev, int device_num) {
    libusb_device_descriptor desc;

    if (libusb_get_device_descriptor(dev, &desc) < 0) return;

    cout << "Device " << device_num << ":" << endl;

    // Основная информация
    cout << "  Class: 0x" << hex << (int)desc.bDeviceClass << " (" << get_device_class_name(desc.bDeviceClass) << ")" << dec << endl;
    cout << "  Vendor ID: 0x" << hex << desc.idVendor << " (" << get_vendor_name(desc.idVendor) << ")" << dec << endl;
    cout << "  Product ID: 0x" << hex << desc.idProduct << " (" << get_product_name(desc.idVendor, desc.idProduct) << ")" << dec << endl;

    // Шина и порт
    uint8_t port = libusb_get_port_number(dev);
    uint8_t bus = libusb_get_bus_number(dev);
    cout << "  Bus: " << (int)bus << ", Port: " << (int)port << endl;

    // Серийный номер
    cout << "  Serial: ";
    print_serial_number(dev);

    // Дополнительная информация
    cout << "  Additional Information:" << endl;
    cout << "    - Subclass: 0x" << hex << (int)desc.bDeviceSubClass << dec << endl;
    cout << "    - Protocol: 0x" << hex << (int)desc.bDeviceProtocol << dec << endl;
    cout << "    - USB Version: " << ((desc.bcdUSB >> 8) & 0xFF) << "."
         << ((desc.bcdUSB >> 4) & 0x0F) << (desc.bcdUSB & 0x0F) << endl;

    cout << endl;
}

int main() {
    libusb_context *ctx = NULL;
    libusb_init(&ctx);
    libusb_set_debug(ctx, 0);

    libusb_device **devs;
    ssize_t cnt = libusb_get_device_list(ctx, &devs);

    if (cnt < 0) {
        libusb_exit(ctx);
        return 1;
    }

    cout << "=== USB DEVICES ===" << endl;
    cout << "Found: " << cnt << " devices" << endl << endl;

    for (int i = 0; i < cnt; i++) {
        print_device_info(devs[i], i + 1);
    }

    libusb_free_device_list(devs, 1);
    libusb_exit(ctx);

    return 0;
}
