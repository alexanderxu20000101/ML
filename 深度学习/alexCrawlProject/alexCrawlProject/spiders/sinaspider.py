# -*- coding: utf-8 -*-
import scrapy


class SinaspiderSpider(scrapy.Spider):
    name = 'sinaspider'
    allowed_domains = ['www.sina.com']
    start_urls = ['http://www.sina.com/']

    def parse(self, response):
        print(response)
